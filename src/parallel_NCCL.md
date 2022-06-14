# Parallel

**parallel.hpp**

```c++
#ifdef USE_NCCL

#include <boost/thread.hpp>

#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/nccl.hpp"

namespace caffe{
    
    // Represents a net parameters. Once a net is created, its parameter buffers can
    // be replaced by ones from Params, to allow parallelization. Params ensures
    // parameters are allocated in one consecutive array.
    template <typename Dtype>
    class Params{
    public:
        explicit Params(shared_ptr<Solver<Dtype>> root_solver);
        virtual ~Params() {}
        
        inline size_t size() const {return size_;}
        inline Dtype* data() const {return data_;}
        inline Dtype* diff() const {return diff_;}
        
    protected:
        const size_t size_; // Size of buffers
        Dtype* data_;		// Network parameters
        Dtype* diff_;		// Gradient
        
    	DISABLE_COPY_AND_ASSIGN(Params);
    };
    
    // Params stored in GPU memory.
    template <typename Dtype>
    class GPUParams: public Params<Dtype>{
    public:
        GPUParams(shared_ptr<Solver<Dtype>> root_solver, int device);
        virtual ~GPUParams();
        
        void Configure(Solver<Dtype*> solver) const;
    protected:
        using Params<Dtype>::size_;
        using Params<Dtype>::data_;
        using Params<Dtype>::diff_;
    };
    
    template <typename Dtype>
    class NCCL: public GPUParams<Dtype>,
    			public Solver<Dtype>::Callback,
    			public Net<Dtype>::Callback{
     public:
         // single process version
         explicit NCCL(shared_ptr<Solver<Dtype>> solver);
         // In multiprocess settings, first create a NCCL id(new_uid), then
         // pass it to each process to create connected instances
         NCCL(shared_ptr<Solver<Dtype>> solver, const string& uid);
         ~NCCL();
         
         // 同步屏障
         boost::barrier* barrier();
         void set_barrier(boost barrier* value);
         // In single process settings, create instances without uids and 
         // call this to connect them
         static void InitSingleProcess(vector<NCCL<Dtype>*>* nccls);
         static string new_uid();
                    
         void Broadcast();
         void RUN(const vector<int>& gpus, const char* restore);
      
     protected:
         void Init();
         void on_start() {}
         void run(int layer);
         void on_gradients_ready();
         
         ncclComm_t comm_;
		 cudaStream_t stream_;
         
         shared_ptr<Solver<Dtype>> solver_;
                    
         boost::barrier* barrier_;
         using Params<Dtype>::size_;
         using Params<Dtype>::data_;
         using Params<Dtype>::diff_;    
     };
    
}

#endif
```

**parallel.cpp**

```c++
#ifdef USE_NCLL

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>

#inlucde "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe{

enum Op{
    copy,
    replace_cpu,
    replace_gpu,
    replace_cpu_diff,
    replce_gpu_diff
};
    
template <typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                         Dtype* buffer, size_t total_size, Op op){
    Dtype* ptr = buffers;
    for(int i=0; i<blob.size(); ++i){
        int size = blob[i]->count();
        switch(op){
            case copy:{
                // Init buffer to current values of blobs
                caffe_copy(size, reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                          ptr);
                break;
            }
            case replace_cpu:
                blobs[i]->data()->set_cpu_data(ptr);
                break;
            case replace_gpu:
                blobs[i]->data()->set_gpu_data(ptr);
                break;
            case replace_cpu_diff:
                blobs[i]->diff()->set_cpu_data(ptr);
                break,
            case replace_gpu_diff:
                blobs[i]->diff()->set_gpu_data(ptr);
                break;
        }
        ptr += size;
    }
    // total_size is at least one byte
    CHECK_EQ(total_size, (ptr==buffer ? 1 : ptr - buffers));
}

// Buffer size necessary to store given blobs
template <typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params){
    size_t size = 0;
    for(int i=0; i<params.size(); ++i)
        size += params[i]->count();
    // Size have at least one byte, otherwise cudaMalloc fails if net has no
  	// learnable parameters.
    return (size > 0) ? size : 1;
}
    
template <typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype>> root_solver )
    : size_(total_size<Dtype>(root_solver->net()->learable_params())), data(), diff(){}
    
template <typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype>> root_solver, int device)
    : Params<Dtype>(root_solver){
        int initial_device;
        CUDA_CHECK(cudaGetDevice(&initial_device));
        
         // Allocate device buffers
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&data_, size_*sizeof(Dtype)));
        
        // Copy blob values
        const vector<Blob<Dtype>*>& net = root_solver->net()->learnable_params();
        apply_buffers(net, data_, size_, copy);
        CUDA_CHECK(cudaMalloc(&diff_, size_*sizeof(Dtype)));
        
        caffe_gpu_set(size_, Dtype(0), diff_);
        
        CUDA_CHECK(cudaSetDevice(initial_device));
}

template <typename Dtype>
GPUParams<Dtype>::~GPUParams(){
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFree(diff_));
}    

template <typename Dtype>
void GPUParams<Dtype>::Configure(Solver<Dtype>* solver) const{
    const vector<Blob<Dtype>*>& net = solver->net()->learnable_params();
    apply_buffers(net, data_, size_, replace_gpu);
    apply_buffers(net, diff_, size_, replace_gpu_diff);
}
    
static int getDevice(){
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}
    
template <typename Dtype>
NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype>> solver)
    : GPUParams<Dtype>(solver, getDevice()), common_(), solver_(solver), barrier_(){
	this->Configure(solver.get());
    Init();
}
    
template <typename Dtype>
NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype>> solver, const string& uid)
    : GPUParams<Dtype>(solver, getDevice()), solver_(solver), barrier_(){
	this->Configure(solver.get());
    Caffe::set_multiprocess(true);
    ncclUniqueId nccl_uid;
    memcpy(&nccl_uid, &uid[0], NCCL_UNIQUE_ID_BYTES);
    NCCL_CHECK(ncclCommInitRank(&common_, Caffe::solver_count,
                               nccl_uid, Caffe::solver_rank)));
   Init();
}
    
template <typename Dtype>
void NCCL<Dtype>::Init(){
    if(solver_->param().layer_wise_reduce()){
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
}
    
template <typename Dtype>
NCCL<Dtype>::~NCCL(){
    if(solver_->param().layer_wise_reduce()){
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    if(comm_){
        ncclCommDestroy(comm_);
    }
}
    
template <typename Dtype>
boost::barrier* NCCL<Dtype>::barrier(){
    return barrier_;
}
    
template <typename Dtype>
void NCCL<Dtype>::InitSingleProcess(vector<NCCL<Dtype>*>* nccls){
     ncclComm_t* comms = new ncclComm_t[nccls->size()]);
    int* gpu_list = new int[nccls->size()];
    for(int i=0; i<nccls->size(); ++i){
        gpu_list[i] = (*nccls)[i]->solver->param().device_id();
    }
    NCCL_CHECK(ncclCommInitAll(comms, static_cast<int>(nccls->size()), gpu_list));
    
    for(int i=0; i<nccls->size(); ++i){
        (*nccls)[i]->common_ = comms[i];
    }
}
    
template <typename Dtype>
string NCCL<Dtype>::new_uid(){
    string uid;
    uid.resize(NCCL_UNIQUE_ID_BYTES);
    
    ncclUniqueId nccl_uid;
    NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
    memcopy(&uid[0], &nccl_uid, NCCL_UNIQUE_ID_BYTES);
    
    return uid;
}
                                  
template <typename Dtype>
void NCCL<Dtype>::Broadcast(){
    if(barrier_){
        barrier_->wait();
    }
    // Copies count values from root to all other devices.
 	// Root specifies the source device in user-order(see ncclCommInit).
 	// Must be called separately for each communicator in communicator clique.
    NCCL_CHECK(ncclBcast(data_, static_cast<int>(size_),
                        nccl::dataType<Dtype>::type, 0,
                        comm_, cudaStreamDefault));
    if(barrier_){
        barrier_->wait();
    }
}
    
template <typename Dtype>
void NCCL<Dtype>::run(int layer){
    CHECK(solver_->param().layer_wise_reduce());
    vector<shared_ptr<Blob<Dtype>>> &blobs = solver_->net()->layers()[layer]->blobs();
 
#ifdef DEBUG
    for(int i=1; i<blobs.size(); ++i){
        CHECK_EQ(blobs[i-1]->gpu_diff() + blobs[i-1]->count(),
                blobs[i+0]->gpu_diff());
    }
#endif
    
    if(blobs.size() > 0){
        // Make sure default stream is done computing gradients. Could be
    	// replaced by cudaEventRecord+cudaStreamWaitEvent to avoid
    	// blocking the default stream, but it's actually slower.
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
        
        int size = 0;
        for(int i=0; i<blobs.size(); ++i){
            size += blobs[i]->count();
        }
        
        if(barrier_){
            barrier_->wait();
        }
        
        // Reduces data arrays of length count in sendbuff using op operation, and leaves
 		// identical copies of result on each GPUs recvbuff.
 		// Sendbuff and recvbuff are assumed to reside on the same device.
 		// Must be called separately for each communicator in communicator clique.
        NCCL_CHECK(ncclAllReduce(blobs[0]->mutable_gpu_diff() /*sendbuff*/,
                                 blobs[0]->mutable_gpu_diff() /*recvbuff*/,
                                 size,
                                 nccl::dataType<Dtype>::type,
                                 ncclSum /*op*/, comm_, stream_));
        caffe_gpu_scal(size, (Dtype)1.0/Caffe::solver_count(),
                       blobs[0]->mutable_gpu_diff(), stream_);
    }
}
    
template <typename Dtype>
void NCCL<Dtype>::on_gradients_ready(){
    if(solver_->param().layer_wise_reduce()){
        CHECK_EQ(solver_->net()->params().size(),
                 solver_->net()->learnable_params().size())
            << "Layer-wise reduce is not supported for nets with shares weights.";
        
        // Make sure reduction is done before applying gradients
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    else{
        if(barrier_){
            barrier_->wait();
        }
        NCCL_CHECK(ncclAllReduce(diff, diff_, static_cast<int>(size_),
                                 nccl::dataType<Dtype>::type, ncclSum, comm_,
                                 cudaStreamDefault));
        caffe_gpu_scal(static_cast<int>(size_),
                       (Dtype) 1.0/Caffe::solver_count(), diff_);
    }
}
    
template <typename Dtype>
class Worker: public InternalThread{
    public:
        explicit Worker(shared_ptr<Solver<Dtype>> rank0, int device,
                        boost::barrier* barrier, vector<NCCL<Dtype>*>* nccls,
                        const char* restore)
            : rank0_(rank0), device_(device), barrier_(barrier),
                nccls_(nccls), restore_(restore){            
        }
        virtual ~Work() {}
    
    protected:
    	void InternalThreadEntry(){
            SolverParameter param(rank0_->param());
            param.set_device_id(device_);
#ifdef DEBUG
         	int device;
            CUDA_CHECK(cudaGetDevice(&device));
            CHECK_EQ(device, device_);
#endif
            param.set_type(rank0_->type());
            shared_ptr<Solver<Dtype>> s(SolverRegistry<Dtype>::CreateSolver(param));
            CHECK_EQ(s->type(), rank0_->type());
            
            if(restore_){
                s->Restore(restore_);
            }
            
            NCCl<Dtype> nccl(s);
            nccl.set_barrier(barrier_);
            s->add_callback(&nccl);
            
            if(s->param().layer_wise_reduce()){
                s->net()->add_after_backward(&nccl);
            }
            
            (*nccl_)[Caffe::solver_rank()] = &nccl;
            // Wait for other threads
            barrier_->wait();
          	// ???
            barrier_->wait();
            // Broadcast rank 0 state
            nccl.Broadcast();
            s->Step(param.max_iter() - s->iter());
            barrier_->wait();
            
 #ifdef DEBUG
            SGDSolver<Dtype>* sa = static_cast<SGDSolver<Dtype>*>(rank0_.get());
            SGDSolver<Dtype>* sb = static_cast<SGDSolver<Dtype>*>(s.get());
            
            for(int h=0; h<sa->history().size(); ++h){
                CUDA_CHECK(cudaSetDevice(sa->param().device_id()));
                const Dtype* a = sa->history()[h]->cpu_data();
                CUDA_CHECK(cudaSetDevice(sb->param().device_id()));
                const Dtype* b = sb->history()[h]->cpu_data();
                for(int v=0; v<sa->history()[h]->count(); ++v){
                    CHECK_DOUBLE_EQ(a[v], b[v]);
                }
            }
 #endif
        }
    
    share_ptr<Solver<Dtype>> rank0_;
    int device_;
    boost::barrier* barrier_;
    vector<NCCL<Dtype>*>* nccls_;
    const char* restore_;
};
    
    template <typename Dtype>
    void NCCL<Dtype>::Run(const vector<int>& gpus, const char* restore){
        boost::barrier barrier(static_cast<int>(gpus.size()));
        vector<NCCL<Dtype>*> nccls(gpus.size());
        // Create workers
        vector<shared_ptr<Worker<Dtype>>> workers(gpus.size());
        for(int i=1; i<gpus.size(); ++i){
            CUDA_CHECK(cudaSetDevice(gpus[i]));
            Caffe::set_solver_rank(i);
            Worker<Dtype>* w = new Worker<Dtype>(solver_, gpus[i], &barrier, &nccls, restore);
            w->StartInternalThread();
            workers[i].reset(w);
        }
        
        CUDA_CHECK(cudaSetDevice(gpus[0]));
        Caffe::set_solver_rank(0);
        barrier_ = &barrier;
        
        solver_->add_callback(this);
        if(solver_->param().layer_wise_reduce()){
            solver_->net()->add_after_backward(this);
        }
        nccls[0] = this;
        // Wait for workers
        barrier.wait();
        // Init NCCL
        InitSingleProcess(&nccls);
        barrier.wait();
        // Run first solver on current thread
        Broadcast();
        solver_->Solver();
        barrier.wait(); // Hangs without it when running tests
        
        // Wait for shutdown
        for(int i=1; i<gpus.size(); ++i){
            workers[i]->StopInternalThread();
        }
    }
    
    INSTANTIATE_CLASS(Params);
    INSTANTIATE_CLASS(GPUParams);
    INSTANTIATE_CLASS(Worker);
    INSTANTIATE_CLASS(NCCL);
  
}
```

 1. **同步屏障：**

    假如有一个很复杂需要很长时间的计算, 但幸运的是, 这个计算可以拆分成几个部分给几个工作线程去计算, 然后再合并结果, 比如多线程版本的排序.

    问题是, 主线程怎么知道工作线程已经完成计算了? `boost::thread::join()`? 这需要这些个工作线程对象是你管理的, 而通常我们都是把任务丢到线程池了, 连线程对象都访问不到, join就更没希望了.

    也许我们可以弄一个count, 再用条件变量联系起来, 主线程初始化了工作线程(或者把worker加入线程池)就去wait这个条件变量, 工作线程完成了就去`count--`, 减到0就`notify`. 就是说, 所有工作线程都完成工作的时候, 主线程会被唤醒来合并结果. 这种操作可以说比较模式化了, 于是人们就将其称为barrier, 通常翻译为”同步屏障”.

    barrier指所有线程都到这个节点, 才能继续往下走. 举个例子, 某公司大门得所有员工打完卡才能开, 这个大门就是一个同步屏障, 大家都等在那. barrier的问题也很明显, 如果有一个员工在上班路上遭遇不幸, 这门就永远打不开了.

