# Common
common中给出的是一些初始化的内容，其中包括随机数生成器的内容以及google的gflags和glog的初始化，其中最主要的还是随机数生成器的内容。

common类在common.hpp和common.cpp中实现。
**common.hpp**

```c++
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>

#include "caffe/util/device_alternate.hpp"

//将宏转换为字符串
#define STRINGIFY(m) #m
#define AS_STRING(m) STRINGIFY(m)

#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif

// 禁止某个类通过构造函数直接初始化另一个类 
// 禁止某个类通过赋值来初始化另一个类
#define DISABLE_COPY_AND_ASSIGN(classname) \
private: \
	classname(const classname&);\
	classname& operation=(const classname&)

#define INSTANTIATE_CLASS(classname) \
	char gInstantiationGuard##classname; \
	template class classname<float>; \
	template class classname<double>

//初始化GPU的前向传播函数
#define INSTANTIATE_LAYER_GPU_FORWARD(classname) \
	template void classname<float>::Backward_gpu(\
		const std::vector<Blob<float>*>& bottom,\
		const std::vector<Blob<float>*>& top);\
	template void classname<double>::Forward_gpu(\
		const std::vector<Blob<double>*>& bottom,\
		const std::vector<Blob<double>*>& top);
//初始化GPU的反向传播函数
#define INSTANTIATE_LAYER_GPU_BACKWARD(classname) \
	template void classname<float>::Backward_gpu(\
		const std::vector<Blob<float>*>& top, \
		const std::vector<bool>& propagate_down, \
		const std::vector<Blob<float>*>& bottom); \
	template void classname<double>::Backward_gpu( \
		const std::vector<Blob<double>*>& top, \
		const std::vector<bool>& propagate_down, \
		const std::vector<Blob<double>*>& bottom)
//初始化GPU的前向反向传播函数
#define INSTANTIATE_LAYER_GPU_FUNCS(classname) \ 
	INSTANTIATE_LAYER_GPU_FORWARD(classname); \
    INSTANTIATE_LAYER_GPU_BACKWARD(classname)
        
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"
        
namespace cv {class Mat;}

namespace caffe{
    
    using boost::shared_ptr;
    using std::fstream;
    using std::ios;
    using std::isnan;
    using std::iterator;
    using std::make_pair;
    using std::map;
    using std::ostringstream;
    using std::pair;
    using std::set;
    using std::string;
    using std::stringstream;
    using std::vector;
    
    // A global initialization function that you should call in your main function.
	// Currently it initializes google flags and google logging.
    void GlobalInit(int* pargc, char*** pargv);
    
    class Caffe{
    public:
        ~Caffe();
        
        // Thread local context for Caffe. Moved to common.cpp instead of
  		// including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  		// on OSX. Also fails on Linux with CUDA 7.0.18.
        //Get函数利用Boost的局部线程存储功能实现
        static Caffe& Get();
        
        enum Brew {CPU, GPU};
        
        // This random number generator facade hides boost and CUDA rng
  		// implementation from one another (for cross-platform compatibility).
        class RNG{
        public:
            RNG(); //利用系统的熵池或者时间来初始化RNG内部的generator_ 
            explicit RNG(unsigned int seed);
            explicit RNG(const RNG&);
            RNG& operator=(const RNG&);
            void* generator();
        
        private:
            class Generator;
            shared_ptr<Generator> generator_;
        };
        
        // Getters for boost rng, curand, and cublas handles
        inline static RNG& rng_stream(){
            if(!Get().random_generator_){
                Get().random_generator_.reset(new RNG());
            }
            return *(Get().random_generator_);
        }
        
#ifndef CPU_ONLY
        inline static cublasHandle_t cublas_handle() {return Get().cublas_handle_;}
        inline static curandGenerator_t curand_generator(){
            return Get().curand_generator_;
        }
#endif
        // Returns the mode: running on CPU or GPU.
        inline static Brew mode() {return Get().mode_;}
        // Sets the mode.
        inline static void set_mode(Brew mode) {Get().mode_ = mode;}
        // Sets the random seed of both boost and curand
        static void set_random_seed(const unsigned int seed);
        // Sets the device. Since we have cublas and curand stuff, set device also
  		// requires us to reset those values.
        static void SetDevice(const int device_id);
        // Prints the current GPU status.
        static void DeviceQuery();
        // Check if specified device is available
        static bool CheckDevice(const int device_id);
        // Search from start_id to the highest possible device ordinal,
  		// return the ordinal of the first available device.
        static int FindDevice(const int start_id = 0);
        
        // Parallel training info
        inline static int solver_count() {return Get().solver_count_;}
        inline static void set_solver_count(int val) {Get().solver_count_ = val;}
        inline static int solver_rank() {return Get().solver_rank_;}
        inline static void set_solver_rank(int val) {Get().solver_rank_ = val;}
        inline static bool multiprocess() {return Get().multiprocess_;}
        inline static void set_multiprocess(bool val) {Get().muktiprocess_ = val;}
        inline static bool root_solver() {return Get().solver_rank_ == 0;}
        
    protected:
#ifndef CPU_ONLY
        cublasHandle_t cublas_handle_; // cublas的句柄  
        curandGenerator_t curand_generator_; // curandGenerator句柄
#endif
        
        shared_ptr<RNG> random_generator_;
        Brew mode_;
        
        int solver_count_;
        int solver_rank_;
        bool multiprocess_;
        
    private:
        Caffe();
        DISABLE_COPY_AND_ASSIGN(Caffe);
    };
}


```

**common.cpp**

```c++
#if defined(_MSC_VER)
#include <process.h>
#include <direct.h>
#define getpid() _getpid()
#endif

#include <boost/thread.hpp>
#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe{
    
    // Make sure each thread can have different values.
	// boost::thread_specific_ptr是线程局部存储机制 
	// 一开始的值是NULL 
    static boost::thread_specific_ptr<Caffe> thread_instance_;
    
    Caffe& Caffe::Get(){
        if(!thread_instance_.get()){ //如果当前线程没有caffe实例
            thread_instance_.reset(new Caffe()); //则新建一个caffe实例并返回
        }
        return *(thread_instance_get());
    }
    
    int64_t cluster_seedgen(void){
        int64_t s, seed, pid;
		
        // 采用传统的基于时间来生成随机数种子
        pid = getpid();
        s = time(NULL);
        seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
        return seed;
    }
    
#ifdef _MSC_VER
    void initGlog(){
      	FLAG_log_dir = ".\\log\\";
      	_mkdir(FLAGS_log_dir.c_str());
        std::string LOG_INFO_FILE;
        std::string LOG_WARNING_FILE;
        std::string LOG_ERROR_FILE;
        std::string LOG_FATAL_FILE;
        
        LOG_INFO_FILE = FLAGS_log_dir + "INFO";
        google::SetLogFilenameExtension(".txt");
        google::SetLogDestination(google::GLOG_INFO, LOG_INFO_FILE.c_tr());
        LOG_WARDNING_FILE = FLAGS_log_dir + "WARNING";
        google::SetLogDestination(google::GLOG_WARNING, LOG_WARDNING_FILE.c_str());
        LOG_ERROR_FILE = FLAGS_log_dir + "ERROR";
        google::SetLogDestination(google::GLOG_ERROR, LOG_ERROR_FILE.c_str());
        LOG_FATAL_FILE = FLAGS_log_dir + "FATAL";
        google::SetLogDestination(google::GLOG_FATAL, LOG_FATAL_.c_str());
    }
#endif
    
    void GlobalInit(int* pargc, char*** pargv){
        ::gfalgs::ParseCommandLineFlags(pargc, pargv);
        
        initGlog();
        ::google::InitGoogleLogging(*(pargv)[0]);
    }
    
    Caffe::Caffe()
        : cublas_handle(NULL), curand_generator_(NULL), random_generator(),
    	mode_(Caffe::CPU), solver_count_(1), solver_rank_(0), multiprocess(false){
            if(cublasCreate(&cublas_handle_) != CUBLAS_STATUS_SUCCESS){
                LOG(ERROR) << "Cannot create Cublas handle. Cublas won't be available.";
            }
            
            if(curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT)
              != CURAND_STATUS_SUCCESS || 
               curandSetPseudoRandomGeneratorSeed(curand_generator_, cluster_seedgen())
              != CURAND_STATUS_SUCCESS){
                LOG(ERROR) << 
                    "Cannot create Curand generator. Curand won't be available.";
            }
        }
    Caffe::~Caffe(){
        if(cublas_handle_) CUBLAS_CHECK(cublasDestroy(cublas_handle_));
        if(curand_generator_){
            CURAND_CHECK(curandDestroyGenerator(curand_generator));
        }
    }
    
    // 初始化CUDA的随机数种子以及cpu的随机数种子
    void Caffe::set_random_seed(const unsigned int seed){
        static bool g_curand_availability_logged = false;
        if(Get().curand_generator_){
            CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(curand_generator(), seed));
            CURAND_CHECK(curandSetGeneratorOffset(curand_generator(), 0));
        }
        else{
            if(!g_curand_availability_logged){
                LOG(ERROR) <<
                    "Curand not available. Skipping setting the curand seed.";
                g_curand_availability_logged = true;
            }
        }
        // RNG seed
        Get().random_generator_.reset(new RNG(seed));
    }
    
    // 设置GPU设备并初始化句柄以及随机数种子 
    void Caffe::SetDevice(const int device_id){
        int current_device;
        CUDA_CHECK(cudaGetDevice(&current_device)); // 获取当前设备id
        if(current_device == device_id){
            return;
        }
        // 在Get之前必须先执行cudasetDevice函数 
        CUDA_CHECK(cudaSetDevice(device_id));
        // 清理以前的句柄
        if(Get().cublas_handle_) CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_));
        if(Get().curand_generator_){
            CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_));
        }
        // 创建新句柄 
        CUBLAS_CHECK(cublasCreate(&Get().cublas_handle));
        CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_,
                                          CURAND_RNG_PSEUDO_DEFAULT));
        // 设置随机数种子 
        CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_,
                                                       cluster_seedgen()));
    }
    
    void Caffe::DeviceQuerry(){
        cudaDeviceProp prop;
        int device;
        if(cudaSuccess != cudaGetDevice(&device)){
            printf("No cuda device present. \n");
            return;
        }
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        LOG(INFO) << "Device id: " << device;
        LOG(INFO) << "Major revision number: " << prop.major;
        LOG(INFO) << "Minor revision number: " << prop.minor;
        LOG(INFO) << "Name: " << prop.name;
        LOG(INFO) << "Total global memory: " << prop.totalGlobalMem;
        LOG(INFO) << "Total shared memory per block: " << prop.sharedMemPerBlock;
        LOG(INFO) << "Total registers per block: " << prop.regsPerBlock;
        LOG(INFO) << "Warp size: " << prop.warpSize;
        LOG(INFO) << "Maximum memory pitch: " << prop.memPitch;
        LOG(INFO) << "Maximum threads per block: " << prop.maxThreadsPerBlock;
        LOG(INFO) << "Maximum dimension of block: " << prop.maxThreadsDim[0] <<
            "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2];
        LOG(INFO) << "Maximum dimension of grid: " << prop.maxGridSize[0] << ","
            << prop.maxGridSize[1] << ","
            << prop.maxGridSize[2];
        LOG(INFO) << "Clock rate: " << prop.clockRate;
        LOG(INFO) << "Total constant memory: " << prop.totalConstMem;
        LOG(INFO) << "Texture alignment: " << prop.textureAlignment;
        LOG(INFO) << "Concurrent copy and execution: "
            << (prop.deviceOverlap ? "Yes" : "No");
        LOG(INFO) << "Number of multiprocessors: " << prop.multiProcessorCount;
        LOG(INFO) << "Kernel execution timeout: "
            << (prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        
        return;
    }
    
    bool Caffe::CheckDevice(const int device_id){
        bool r = ((cudaSuccess == cudaSetDevice(device_id)) &&
                 (cudaSuccess == cudaFree(0)));
        cudaGetLastError();
        return r;
    }
    
    int Caffe::FindDevice(const int start_id){
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        for(int i = start_id; i < count; i++){
            if(CheckDevice(i)) return i;
        }
        return -1;
    }
    
    class Caffe::RNG::Generator{
    public:
        Generator() : rng_(new caffe::rng_t(cluster_seedgen())){}
        explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)){}
        caffe::rng_t* rng() {return rng_.get();} 
    private:
        shared_ptr<caffe::rng_t> rng_;
    };
    
    Caffe::RNG::RNG() : generator_(new Generator()) {}
    Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}
    Caffe::RNG& Caffe::RNG::operator=(const RNG& other){
        generator_.reset(other.generator_.get());
        return *this;
    }
    
    void* Caffe::RNG::generator(){
        return static_cast<void*>(generator_->rng());
    }
    
    const char* cublasGetErrorString(cublasStatus_t error){
        switch(error){
            case CUBLAS_STATUS_SUCCESS:
                return "CUBLAS_STATUS_SUCCESS";
            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";
            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";
            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";
            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";
            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";
            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";
            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
        }
        return "Unkonwn cublas status";
    }
    
    const char* curandGetErrorString(curandStatus_t error){
        switch(error){
            case CURAND_STATUS_SUCCESS:
                return "CURAND_STATUS_SUCCESS";
            case CURAND_STATUS_VERSION_MISMATCH:
                return "CURAND_STATUS_VERSION_MISMATCH";
            case CURAND_STATUS_NOT_INITIALIZED:
                return "CURAND_STATUS_NOT_INITIALIZED";
            case CURAND_STATUS_ALLOCATION_FAILED:
                return "CURAND_STATUS_ALLOCATION_FAILED";
            case CURAND_STATUS_TYPE_ERROR:
                return "CURAND_STATUS_TYPE_ERROR";
            case CURAND_STATUS_OUT_OF_RANGE:
                return "CURAND_STATUS_OUT_OF_RANGE";
            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
                return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
            case CURAND_STATUS_LAUNCH_FAILURE:
                return "CURAND_STATUS_LAUNCH_FAILURE";
            case CURAND_STATUS_PREEXISTING_FAILURE:
                return "CURAND_STATUS_PREEXISTING_FAILURE";
            case CURAND_STATUS_INITIALIZED_FAILED:
                return "CURAND_STATUS_INITIALIZED_FAILED";
            case CURAND_STATUS_ARCH_MISMATCH:
                return "CURAND_STATUS_ARCH_MISMATCH";
            case CURAND_STATUS_INTERNAL_ERROR:
                return "CURAND_STATUS_INTERNAL_ERROR";
        }
        return "Unknown curand status";
    }
}
```



