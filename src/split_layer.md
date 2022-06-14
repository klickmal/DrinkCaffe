# Split Layer
split layer将输入复制到多个输出中：

1. 在正向传播时，不需要将输入拷贝到输出中，输出块直接共享输入内存即可，也就是仅复制指针
2. 在反向传播时，每一个输出blob都有自己的diff，所有输出的diff按元素进行相加即可


split layer在源文件split_layer.hpp, split_layer.cpp和split_layer.cu中实现

**split_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
	template <typename Dtype>
	class SplitLayer: public Layer<Dtype>{
    public:
        explicit SplitLayer(const LayerParameter& param):
            Layer<Dtype>(param){}
    	
    	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    	
        virtual inline const char* type() const {return "Split";} 
    	virtual inline int ExactNumBottomBlobs() const {return 1;}
    	virtual inline int MinTopBlobs() const {return 1;}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
		// 反向传播, 其中propagate_down表示每一个bottom块是否需要求梯度。
		// bottom_diff = top[1]_diff + top[1]_diff + ... + top[n]_diff.
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype*>& bottom);
    	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<Blob<Dtype>*>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
    	int count_;
    };
}
```

**split_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void SplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top){
        count_ = bottom[0]->count();
        
        for(int i=0; i < top.size(), ++i){
            CHECK_NE(top[i], bottom[0]) << this->type() << "Layer does not"
                "allow in-place computation."
            top[i]->ReshapeLike(*bottom[0]);
            CHECK_EQ(count_, top[i]->count());
        }
    }
    
    template <typename Dtype>
    void SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
        for(int i=0; i<top.size(); ++i){
            top[i]->ShareData(*bottom[0]);
        }
    }
    
    template <typename Dtype>
    void SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            if(top.size()==1){
                caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
            	return;
            }
            
            caffe_add(count, top[0]->cpu_diff(), top[1]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  			for(int i=2; i<top.size(); ++i){
                const Dtype* top_diff = top[i]->cpu_diff();
                Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
                caffe_axpy(count, Dtype(1), top_diff, bottom_diff);
            }
        }
    }

#ifdef CPU_ONLY
	STUP_GPU(SplitLayer);
#endif
    
    INSTANTIATE_CLASS(SplitLayer);
    REGISTER_LAYER_CLASS(Split);
}

```

 1. caffe_copy在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_copy(const int N, const Dtype *X, Dtype *Y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <typename Dtype>
    	void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
    		if (X != Y) {
    			if (Caffe::mode() == Caffe::GPU) {
    #ifndef CPU_ONLY
    				// NOLINT_NEXT_LINE(caffe/alt_fn)
    				CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
    #else
    				NO_GPU;
    #endif
    		}
    			else {
    				memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    			}
    	}
    }
    
    template void caffe_copy<int>(const int N, const int* X, int* Y);
    template void caffe_copy<unsigned int>(const int N, const unsigned int* X, unsigned int* Y);
    template void caffe_copy<float>(const int N, const float* X, float* Y);
    template void caffe_copy<double>(const int N, const double* X, double* Y);
    ```

 2. caffe_add在math_functions.hpp声明：

    ```
    template <typename Dtype>
    void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    caffe_add在math_functions.cpp中实现如下：

    ```c++
    template <>
    void caffe_add<float>(const int n, const float* a, const float* b, float* y) {
    	vsAdd(n, a, b, y);
    }
    
    template <>
    void caffe_add<double>(const int n, const double* a, const double* b, double* y) {
    	vdAdd(n, a, b, y);
    }
    ```

    ```c++
    #define DEFINE_VSL_BINARY_FUNC(name, operation) \
      template<typename Dtype> \
      void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
        CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
        for (int i = 0; i < n; ++i) { operation; } \
      } \
      inline void vs##name( \
        const int n, const float* a, const float* b, float* y) { \
        v##name<float>(n, a, b, y); \
      } \
      inline void vd##name( \
          const int n, const double* a, const double* b, double* y) { \
        v##name<double>(n, a, b, y); \
      }
    
    DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
    DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
    DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
    DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])
    ```

 3. caffe_axpy函数在math_functions.hpp中声明：

    ```c++
    template <typename Dtype>
    void caffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
    ```

    caffe_axpy在math_functions.cpp实现如下：

    ```c++
    template <>
    void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) {
    	cblas_saxpy(N, alpha, X, 1, Y, 1);
    }
    
    template <>
    void caffe_axpy<double>(const int N, const double alpha, const double* X, double* Y) {
    	cblas_daxpy(N, alpha, X, 1, Y, 1);
    }
    ```

    cblas_saxpy、cblas_daxpy在OpenBLAS接口定义如下，其功能如下式：
    $$
    Y=alpha * X +beta*Y
    $$

    ```c++
    void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
    void cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
    ```

**split_layer.cu**

```c++
#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

  	template <typename Dtype>
    void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        for(int i=0; i<top.size(); ++i){
            top[i]->ShareData(*bottom[0]);
        }
    }
    
    template <typename Dtype>
    void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            if(top.size()==1){
                caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_data());
                return;
            }
            
            caffe_gpu_add(count_, top[0]->gpu_diff(), top[1]->gpu_diff(), bottom[0]->mutable_gpu_diff());
        	
            for(int i=2; i<top.size(); ++i){
                const Dtype* top_diff = top[i]->gpu_diff();
                Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
                caffe_gpu_axpy(count_, Dtype(1), top_diff, bottom_diff);
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);
}
```

1. caffe_copy在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_copy(const int N, const Dtype *X, Dtype *Y);
   ```

   caffe_copy在math_functions.hpp实现如下：

   ```c++
   template <typename Dtype>
   void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
   	if (X != Y) {
   		if (Caffe::mode() == Caffe::GPU) {
   #ifndef CPU_ONLY
   			// NOLINT_NEXT_LINE(caffe/alt_fn)
   			CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
   #else
   			NO_GPU;
   #endif
   		}
   		else {
   			memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
   		}
   	}
   }
   
   template void caffe_copy<int>(const int N, const int* X, int* Y);
   template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
   		unsigned int* Y);
   template void caffe_copy<float>(const int N, const float* X, float* Y);
   template void caffe_copy<double>(const int N, const double* X, double* Y);
   ```

2. caffe_gpu_add在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);
   ```

   caffe_gpu_add在math_functions.cu实现如下：

   ```c++
   template <typename Dtype>
   __global__ void add_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
   	CUDA_KERNEL_LOOP(index, n) {
   		y[index] = a[index] + b[index];
   	}
   }
   
   template <>
   void caffe_gpu_add<float>(const int N, const float* a, const float* b, float* y) {
   	// NOLINT_NEXT_LINE(whitespace/operators)
   	add_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
   }
   
   template <>
   void caffe_gpu_add<double>(const int N, const double* a, const double* b, double* y) {
   	// NOLINT_NEXT_LINE(whitespace/operators)
   	add_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
   }
   ```

3. caffe_gpu_axpy在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
       Dtype* Y);
   ```

   caffe_gpu_axpy在math_functions.cu实现如下：

   ```c++
   template <>
   void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X, float* Y) {
   	CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
   }
   
   template <>
   void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X, double* Y) {
   	CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
   }
   ```

   ```c++
   CUBLASAPI cublasStatus_t CUBLASWINAPI cublasSaxpy_v2 (cublasHandle_t handle,
                                                         int n, 
                                                         const float *alpha, /* host or device pointer */
                                                         const float *x, 
                                                         int incx, 
                                                         float *y, 
                                                         int incy);
   
   CUBLASAPI cublasStatus_t CUBLASWINAPI cublasDaxpy_v2 (cublasHandle_t handle,
                                                         int n, 
                                                         const double *alpha, /* host or device pointer */
                                                         const double *x, 
                                                         int incx, 
                                                         double *y, 
                                                         int incy);
   ```

   
