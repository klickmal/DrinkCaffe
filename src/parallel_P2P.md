# Parallel

**parallel.hpp**

```c++
#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"

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
    
    class DevicePair{
    public:
        DevicePair(int parent, int device): parent_(parent), device_(device){}
        
        inline int parent() {return parent_;}
        inline int device() {return device_;}
        
        // Group GPUs in pairs, by proximity depending on machine's topology
        static void compute(const vector<int> devices, vector<DevicePair>* pairs);
        
    protected:
        int parent_;
        int device_;
    };
    
    // Synchronous data parallelism using map-reduce(映射归约) between local GPUS
    // P2PSync主要是在多个本地GPU之间使用map-reduce来同步数据，该类继承于GPUParams和Solver的Callback。
    // 其中GPUParams是管理存放在GPU端的数据，Callback使solver训练中可以用回调的方式来调用P2PSync相关函数来进行数据同步操作。
    template <typename Dtype>
    class P2PSync: public GPUParams<Dtype>, public Solver<Dtype>::Callback, public InternalThread
    {
    public:
        explicit P2PSync(shared_ptr<Solver<Dtype>> root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param);
        virtual ~P2PSync();
        
        inline const shared_ptr<Solver<Dtype>>& solver() const {return solver_;}
        
        // 运行，树的构建、开线程，开始训练
        void Run(const vector<int>& gpus);
        // 对GPU分组，并构建GPU树
        void Prepare(const vector<int>& gpus, vector<shared_ptr<P2PSync<Dtype>>>* syncs);
        inline int initial_iter() const {return initial_iter_;}
        
    protected:
        // 在solve中进行前向之前，用于同步网络参数给子GPU
        void on_start();
        // 将子GPU的梯度汇总到主GPU
        void on_gradients_ready();
        // 内部线程入口
        void InternalThreadEntry();
        
        P2PSync<Dtype>* parent_;
        vector<P2PSync<Dtype>*> children_;
        BlockingQueue<P2PSync<Dtype>*> queue_;
        
        const int initial_iter_;
        Dtype* parent_grads_;
        shared_ptr<Solver<Dtype>> solver_;
        
        // 网络参数和梯度是需要跨GPU更新同步的。
        using Params<Dtype>::size_;
        using Params<Dtype>::data_;
        using Params<Dtype>::diff_;
    };
}
```

**parallel.cpp**

```c++
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"

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
    
    void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs){
    #ifndef
        vector<int> remaining(devices);
        
        // depth for reduction tree
        int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));
        
        // Group GPUs by board
        for(int d = 0; d < remaining_depth; ++d){
            for(int i = 0; i < remaining.size(); ++i){
                for(int j = i + 1; j < remaining.size(); ++j){
                    cudaDeviceProp a, b;
                    CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]));
                    CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]));
                    
                    if(a.isMultiGpuBoard && b.isMultiGpuBoard){
                        if(a.multiGpuBoardGroupID == b.multiGpuBoardGroupID){
                            pairs->push_back(DevicePair(remaining[i], remaining[j]));
                            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
                            remaining.erase(remaining.begin() + j);
                        }
                    }
                }
            }
        }
        
        ostringstream s;
        for(int i = 0; i < remaining.size(); ++i){
            s << (i ? ", " : " ") << remaining[i];
        }
        DLOG(INFO) << "GPUs paired by boards, remaining:" << s.str();
        
        // Group by P2P accessibility
        remaining_depth = ceil(log2(remaining.size()));
        for(int d = 0; d < remaining_depth; ++d){
            for(int i = 0; i < remaining.size(); ++i){
                pairs->push_back(DevicePair(remaining[i], remaining[i+1]));
                DLOG(INFO) << "Remaining pair: " << remaining[i] << ":" << remaining[i+1];
                remaining.erase(remaining.begin() + i + 1);
            }
        }
        
        // Should only be the parent node remaining
        CHECK_EQ(remaining.size(), 1);
        
        pairs->insert(pairs->begin(), DevicePairs(-1, remaining[0]));
        
        CHECK(pairs->size() == devices.size());
        for(int i = 0; i < pairs.size(); ++i){
            CHECK((*pairs)[i].parent() != (*pairs)[i].device());
            
            for(int j = i+1; j < pairs->size(); ++j){
                CHECK((*pairs)[i].device() != (*pairs)[j].device());
            }
        }
        
    #else
        NO_GPU;
    #endif
    }
    
    template <typename Dtype>
    P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype>> root_solver,
                           P2PSync<Dtype>* parent, const SolverParameter& param)
        : GPUParams<Dtype>(root_solver, param.device_id()),
    	  parent_(parent),  children_(), queue_(), initial_iter_(root_solver_iter()), solver_(){
    #ifndef CPU_ONLY
    	int initial_device;
        CUDA_CHECK(cudaGetDevice(&initial_device));
        const int self = param.device_id();
        CUDA_CHECK(cudaSetDevice(self));
              
        if(parent == NULL){
            solver_ = root_solver;
        }
        else{
            Caffe::set_root_solver(false);
           	solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
            Caffe::set_root_solver(true);
            Caffe::set_root_solver_ptr(root_solver.get());
        }
        this->configure(solver_.get());
        solver_->add_callback(this);
        
        if(parent){
            const int peer = parent->solver_->param().device_id();
            int access;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&accsess, self, peer));
            if(access){
                CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
            }
            else{
                LOG(INFO) << "GPU " << self << " does not have p2p access to GPU " << peer;
            }
            
            CUDA_CHECK(cudaSetDevice(peer));
            CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
            CUDA_CHECK(cudaSetDevice(self));
        }
              
         CUDA_CHECK(cudaSetDevice(initial_device));
              
    #else
         NO_GPU;
    #endif
    }
    
    template <typename Dtype>
    P2PSync<Dtype>::~P2PSync(){
    #ifndef CPU_ONLY
        int initial_device;
        CUDA_CHECK(cudaGetDevice(&initial_device));
        const int peer = parent_->solver_->param().device_id();
        CUDA_CHECK(cudaSetDevice(self));
        
        if(parent_){
            CUDA_CHECK(cudaFree(parent_grads_));
            const int peer = parent_->solver_->param().device_id();
            
            int access;
        	CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
        	if(access){
            	CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
        	}
        }
        CUDA_CHECK(cudaSetDevice(initial_device));
    #endif
    }
    
    template <typename Dtype>
    void P2PSync<Dtype>::InternalThreadEntry(){
        Caffe::SetDevice(solver_->param().device_id());
        CHECK(Caffe::root_solver());
        Caffe::set_root_solver(false);
        
        if(solver_->param().random_seed() >=0 ){
            Caffe::set_random_seed(solver_->param().random_seed() + solver_->param().device_id());
        }
        solver_->Step(solver_->param().max_iter() - initial_iter_);
    }
	
    template <typename Dtype>
    void P2PSync<Dtype>::on_start(){
    #ifndef CPU_ONLY
    #ifdef DEBUG
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CHECK(device == solver_->param().device_id());
    #else
   	#endif
    	if(parent_){
     		P2PSync<Dtype>* parent = queue_.pop();
            CHECK(parent == parent_);
        }   
        for(int i=children_.size()-1; i>=0; i--){
            Dtype* src = data_;
            Dtype* dst = children_[i]->data_;
 
        #ifdef DEBUG
            cudaPointerAttributes attributes;
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
            CHECK(attributes.device == device);
            CUDA_CHECK(cudaPointerGetAttributes(&atrributes, dst));
            CHECK(attributes.device == children_[i]->solver_->param().device_id());
        #endif

            CUDA_CHECK(cudaMemcpyAsync(dst, src, size_*sizeof(Dtype), 
                                       cudaMemcpyDeviceToDevice, cudaStreamDefault));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
            children_[i]->queue_.push(this);
    	}
        #endif
    }
    
    // on_gradients_ready()函数，该函数分为两个部分：
    // 第一部分是多个GPU的梯度加和，第二部分是将计算后的梯度传给根GPU(GPU0)。第一部分的代码如下：
    template <typename Dtype>
    void P2PSync<Dtype>::on_gradients_ready(){
    #ifndef CPU_ONLY
  	#ifdef DEBUG
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        CHECK(device == solver_->param().device_id());
    #endif
        
        // Sum children gradients as they appear int the queue
        for(int i=0; i<children.size(); ++i){
        	P2PSync<Dtype>* child = queue_.pop();
            Dtype* src = child->parent_grads_;
            Dtype* dst = diff_;
            
        	caffe_gpu_add(size_, src, dst, dst);
        }
        
        if(parent_){
            Dtype* src = diff_;
            Dtype* dst = parent_grads_;
            
            CUDA_CHECK(cudaMemcpyAsync(dst, src, size_*sizeof(Dtype), 
                                       cudaMemcpyDeviceToDevice, cudaStreamDefault));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
            parent_->queue_.push(this);
        }
        else{
            caffe_gpu_scal(size_, Dtype(1.0/Caffe::solver_count()), diff_);
        }
    #endif
    }
    
    template <typename Dtype>
    void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
                                vector<shared_ptr<P2PSync<Dtype>>>* syncs){
        // pair devices for map-reduce synchronization
        vector<DevicePair> pairs;
        DevicePair::comnpute(gpus, &pairs);
        
        ostringstream s;
        for(int i=1; i<pairs.size(); ++i){
            s << (i==1 ? "":",") << pairs[i].parent() << ":" << pairs[i].device();
        }
        
        LOG(INFO) << "GPUs pairs " << s.str();
        
        SolverParameter param(solver_->param());
        
        // Build the GPU tree by finding the parent for each solver
        for(int attempts = 0; attempts < pairs.size(); ++attempts){
            for(int i=1; i<pairs.size(); ++i){
                if(!syncs->at(i).get()){
                    P2PSync<Dtype>* parent = NULL;
                    for(int j=0; j<syncs->size(); ++j){
                        P2PSync<Dtype>* parent = NULL;
                        for(int j=0; j<syncs->size(); ++j){
                            P2PSync<Dtype>* sync = j==0 ? this : syncs->at(j).get();
                            if(sync){
                                const SolverParameter& p = sync->solver()->param();
                                if(p.device_id() == pairs[i].parent()){
                                    parent = sync;
                            }
                         }
                     }
                     if(parent){
                     	param.set_device_id(pairs[i].device());
                       	syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
                       	parent->children_.push_back((P2PSync<Dtype>*)syncs->at(i).get());
                    }
              	}
            }
        }
    }
    
    template <typename Dtype>
    void P2PSync<Dtype>::Run(const vector<int>& gpus){
        vector<shared_ptr<P2PSync<Dtype>>> syncs(gpus.size());
        Prepare(gpus, &syncs);
        
        LOG(INFO) << "Starting Optimization";
        
        for(int i=1; i<syncs.size(); ++i){
            syncs[i]->StartInternalThread();
        }
        
        // Run root solver on current thread
        solver_->Solve();
        
        for(int i=1; i<syncs.size(); ++i){
            syncs[i]->StopInternalThread();
        }
    }
    
    INSTANTIATE_CLASS(Params);
    INSTANTIATE_CLASS(GPUParams);
    INSTANTIATE_CLASS(P2PSync);
}
```

 1. 

