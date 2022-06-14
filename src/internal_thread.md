# Internal Thread
该类实际是boost::thread的包裹器。boost十分强大，可以不用修改就在linux和windows使用，避免了使用内核函数的移植性问题。

Internal thread在internal_thread.hpp和internal_thread.cpp中实现

**internal_thread.hpp**

```c++
#include "caffe/common.hpp"

namespace boost {class thread;}

namespace caffe{
    
    class InternalThread{
    public:
        // 构造函数默认初始化boost::thread
        InternalThread(): thread_() {}
        // 析构函数中调用停止线程函数
        virtual ~InternalThread();
        
        // 开始线程
        void StartInternalThread();
        // 停止线程
        void StopInternalThread();
        // 判断线程是否开始
        bool is_started() const;
     
    protected:
        // Implement this method in your subclass
      	// with the code you want your thread to run.
        // 定义一个虚函数，继承类可以进行相关实现
        virtual void InternalThreadEntry() {}
        // Should be tested when running loops to exit when requested.
        // 在当请求退出的时候应该调用该函数  
        bool must_stop();
    
   	private:
        // For NCCL
        // void entry(int device, Caffe::Brew mode, int rand_seed,
                 // int solver_count, int solver_rank, bool multiprocess);
        
        // multi-GPU
        void entry(int device, Caffe::Brew mode, int rand_seed, 
                   int solver_count, bool root_solver);
        
        shared_ptr<boost::thread> thread_;
    };
}
```

**internal_thread.cpp**

```c++
#include <boost/thread.hpp>
#include <exception>

#include "caffe/internal_thread.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    // 析构函数，调用停止内部线程函数  
	InternalThread::~InternalThread(){
        StopInternalThread();
    }
    // 测试线程是否起来  
    bool InternalThread::is_started() const {
        return thread_ && thread_->joinable(); // 首先thread_指针不能为空，然后该线程是可等待的（joinable） 
    }
    
    // 如何结束线程？
    // 要想主动终结一个线程是很费劲的，caffe也没有做这种费力不讨好的事情，而是被动的等待。
    // caffe采取是设置must_stop()函数自动检测终结条件的退出方式。
    // thread里的thread_->interruption_requested()提供中断检测。
    bool InternalThread::must_stop(){
        // true if interruption has been requested for the current thread, false otherwise. 
        return thread_ && thread_->interruption_requested();
    }
    
    void InternalThread::StartInternalThread(){
        CHECK(!is_started()) << "Threads should persist and not be restarted.";
        
        int device = 0;
#ifndef CPU_ONLY
        CUDA_CHECK(cudaGetDevice(&device));
#endif
        Caffe::Brew mode = Caffe::mode();
        int rand_seed = caffe_rng_rand();
        int solver_count = Caffe::solver_count();
        // int solver_rank = Caffe::solver_rank();
        // bool multiprocess = Caffe::multiprocess();
        bool root_solver = Caffe::root_solver();
        
        try{// 重新实例化一个thread对象给thread_指针，该线程的执行的是entry函数
            //thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
              //                      rand_seed, solver_count, solver_rank, multiprocess);
			thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
                                           rand_seed, solver_count, root_solver));
        }
        catch(std::exception& e){
            LOG(FATAL) << "Thread exception: " << e.what();
        }
    }
    
   	// 线程所要执行的函数
    // void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
       //                       int solver_count, int solver_rank, bool multiprocess){
//#ifndef CPU_ONLY
  //      CUDA_CHECK(cudaSetDevice(device));
//#endif
       // Caffe::set_mode(mode);
       // Caffe::set_random_seed(rand_seed);
       // Caffe::set_solver_count(solver_count);
       // Caffe::set_solver_rank(solver_rank);
       // Caffe::set_multiprocess(multiprocess);
        
       // InternalThreadEntry();
    // }
    
    // 线程所要执行的函数
    void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
                              int solver_count, bool root_solver){
    #ifndef CPU_ONLY
        CUDA_CHECK(cudaSetDevice(device));
    #endif
        Caffe::set_mode(mode);
        Caffe::set_random_seed(rand_seed);
        Caffe::set_solver_count(solver_count);
        Caffe::set_root_solver(root_solver);
        
        InternalThreadEntry();
    }
    
    // 停止线程
    void InternalThread::StopInternalThread(){
        if(is_started()){ // 如果线程已经开始
            // A running thread can be interrupted by invoking the interrupt() 
            // member function of the corresponding boost::thread object. 
            thread_->interrupt(); // 那么打断
            try{
                // In order to wait for a thread of execution to finish, the join(), 
                // __join_for or __join_until ( timed_join() deprecated) member functions of the 	   						// boost::thread object must be used
                thread_->join(); // 等待线程结束
            }
            catch(boost::thread_interrupted&){} // //如果被打断，啥也不干，因为是自己要打断的^_^
            catch(std::exception& e){// 如果发生其他错误则记录到日志
                LOG(FATAL) << "Thread exception: " << e.what();
            }
        }
    }                      
}
```



