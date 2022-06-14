# Syncedmem
Syncedmem在syncedmem.hpp和syncedmem.cpp中实现
**syncedmem.hpp**

```c++
#include <cstdlib>

#ifdef USE_MKL
#include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace  caffe{
    
    inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda){
#ifndef CPU_ONLY
        if(Caffe::mode() == Caffe::GPU){
            CUDA_CHECK(cudaMallocHost(ptr, size));
            *use_cuda = true;
            return;
        }
#endif
#ifdef USE_MKL
        *ptr = mkl_malloc(size ? size : 1, 64);
#else
        *ptr = malloc(size);
#endif
        *use_cuda  = false;
        CHECK(*ptr) << "host allocation of size " << size << " failed";
    }
    
    inline void Caffe_FreeHost(void* ptr, bool* use_cuda){
#ifndef CPU_ONLY
        if(use_cuda){
            cudaFreeHost(ptr);
            return;
        }
#endif
#ifdef USE_MKL
        mkl_free(ptr);
#else
        free(ptr);
#endif
    }
    
    // Manages memory allocation and synchronization between the host (CPU) and device (GPU).
    class SynedMemory{
    public:
        SyncedMemory();
        
        // 带explicit关键字：单个参数构造函数，explicit禁止单参数构造函数的隐式转换
        explicit SyncedMemory(size_t size);
        
        ~SyncedMemory();
        const void* cpu_data();  //获得cpu上data的地址
        void set_cpu_data(void* data); //将cpu的data指针指向一个新的区域由data指针传入，并且将原来申请的内存释放。
        const void* gpu_data(); //获得gpu数据地址
        void set_gpu_data(void* data); //将gpu的data指针指向一个新的区域由data指针传入，并且将原来申请的内存释放。
        void* mutable_cpu_data(); 
        void* mutable_gpu_data();
        
        //共享内存的4种状态：未初始化，CPU数据有效，GPU数据有效，已同步
        enum SyncedHead {UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED}; 
        SyncedHead head() {return head_;} // 返回当前共享内存的状态
        size_t size() {return size_;}

#ifndef CPU_ONLY
        void async_gpu_push(const cudaStream_t& stream);
#endif
        
    private:
        void check_device();
        
        void to_cpu();
        void to_gpu();
        void* cpu_ptr_;
        void* gpu_ptr_;
        
        size_t size_;
        SyncedHead head_;
        bool own_cpu_data_;
        bool cpu_malloc_use_cuda_; //分配cpu内存是否用cudaMallocHost()分配
        bool own_gpu_data_;
        int device_;
        
        DISABLE_COPY_AND_ASSIGN(SyncedMemory);
    };
}
```

**syncedmem.cpp**

```c++
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	SyncedMemory::SyncedMemory()
        : cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false){
#ifndef CPU_ONLY
#ifdef DEBUG
                CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
    }
   	
     SyncedMemory::SyncedMemory(size_t size)
         : cpu_ptr(NULL), gpu_ptr(NULL), size_(size), head_(UNINITIALIZED),
    		own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false){
#ifndef CPU_ONLY
#ifdef DEBUG
                CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
    }
    
    SyncedMemory::~SyncedMemory(){
        check_device();
        if(cpu_ptr_ && own_cpu_data_){
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
#ifndef CPU_ONLTY
        if(gpu_ptr_ && own_gpu_data_){
            cudaFree(gpu_ptr);
        }
#endif
    }
    
    inline void SyncedMemory::to_cpu(){
        check_device();
        switch(head_){
            case UNINITIALIZED:
                CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
                caffe_memset(size_, 0, cpu_ptr_);
                head_ = HEAD_AT_CPU;
                own_cpu_data_ = true;
                break;
            case HEAD_AT_GPU:
#ifdef CPU_ONLY
            	if(cpu_ptr_ == NULL){
                    CaffeMallocHost(&cpu_ptr_, size, &cpu_malloc_use_cuda_);
                    own_cpu_data_ = true;
                }
                caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
                head_ = SYNCED;
#else
                NO_GPU;
#endif
                break;
            case HEAD_AT_CPU:
            case SYNCED:
                break;
        }
    }
    
    inline void SyncedMemory::to_gpu(){
        check_device();
#ifndef CPU_ONLY
        switch(head_){
            case UNINITIALIZED:
                CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                caffe_gpu_memset(size_, 0, gpu_ptr_);
                head_ = HEAD_AT_GPU;
                own_gpu_data_ = true;
                break;
            case HEAD_AT_CPU:
                if(gpu_ptr_ == NULL){
                    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
                    own_gpu_data_ = true;
                }
                caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
                head_ = SYNCED;
                break;
            case HEAD_AT_GPU:
            case SYNCED:
                break;
        }
#else
        NO_GPU;
#endif
    }
    
    const void* SyncedMemory::cpu_data(){
        check_device();
        to_cpu();
        return (const void*)cpu_ptr_;
    }
    
    void SyncedMemory::set_cpu_data(void* data){
        check_device();
        CHECK(data);
        if(own_cpu_data_){
            CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
        }
        cpu_ptr_ = data;
        head_ = HEAD_AT_CPU;
        own_cpu_data_ = false;
    }
    
    const void* SyncedMemory::gpu_data(){
        check_device();
#ifndef CPU_ONLY
        to_gpu();
        return (const void*)gpu_ptr_;
#else
        NO_GPU;
        return NULL;
#endif
    }
    
    void SyncedMemory::set_gpu_data(void* data){
        check_device();
#ifndef CPU_ONLY
        CHECK(data);
        if(own_gpu_data_){
            CUDA_CHECK(cudaFree(gpu_ptr_));
        }
        gpu_ptr_ = data;
        head_ = HEAD_AT_GPU;
        own_gpu_data_ = false;
#else
        NO_GPU;
#endif
    }
    
    void* SyncedMemory::mutable_cpu_data(){
        check_device();
        to_cpu();
        head_ = HEAD_AT_CPU();
        return cpu_ptr_;
    }
    
    void SyncedMemory::mutable_gpu_data(){
        check_device();
#ifndef CPU_ONLY
        to_gpu();
        head_ = HEAD_AT_GPU;
        return gpu_ptr_;
#else
        NO_GPU;
        return NULL;
#endif
    }
    
#ifndef CPU_ONLY
    // cuda中的流同步，这里传入一个异步流，在计算的时候向GPU复制数据。
    void SyncedMemory::async_gpu_push(const cudaStream_t& stream){
        check_device();
        CHECK(head_ == HEAD_AT_CPU);
        if(gpu_ptr_ == NULL){
            CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
            own_gpu_data_ = true;
        }
        
        const cudaMemcpyKind put = cudaMemcpyHostToDevice;
        CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
        head_ = SYNCED;
    }
#endif
    
    void SyncedMemory::chenck_device(){
#ifndef CPU_ONLY
#ifdef DEBUG
        int device;
        cudaGetDevice(&device);
        CHECK(device == device_);
        if(gpu_ptr_ && own_gpu_data_){
            cudaPointerAttributes attributes;
            CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
            CHECK(attributes.device == device_);
        }
#endif
#endif
    }
  
}
```

1. caffe_memset在math_functions.hpp中声明定义如下：

   ```c++
   inline void caffe_memset(const size_t N, const int alpha, void* X) {
     memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
   }
   ```

2. caffe_gpu_memory在math_functions.hpp声明如下：

   ```c++
   void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);
   ```

   在math_functions.cu中定义如下：

   ```c++
   void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
   	if (X != Y) {
   		CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
   	}
   }
   ```

   

