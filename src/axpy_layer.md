# Axpy Layer

Axpy layer实现channel-wise scale和element-wise add操作。计算公式如下：

F = a * X + Y

 * Shape info:

   a:  N x C                 --> bottom[0]      

   X:  N x C x H x W  --> bottom[1]       

   Y:  N x C x H x W  --> bottom[2]     

   

   F:  N x C x H x W  --> top[0]

   

axpy layer在源文件**axpy_layer.hpp、axpy_layer.cpp、axpy_layer.cu**中实现。

**axpy_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namesapce caffe{
	template <typename Dtype>
    class AxpyLayer: public Layer<Dtype>{
    public: 
      	explicit AxpyLayer(const LayerParameter& param):
        	Layer<Dtype>(param){}
        
      	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                              const vector<Blob<Dtype>*>& top){}
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                            const vector<Blob<Dtype>*>& top);
        
      	virtual inline const char* type() const {return "Axpy";}
      	virtual inline int ExactNumBottomBlobs() const {return 1;}
      	virtual inline int EsxactNumTopBlobs() const {return 1;}
     
     protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
                                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      							const vector<Blob<Dtype>*>& top);
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      							const vector<bool>& propagate_down, 
                                const vector<Blob<Dtype>*>& bottom);
  		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      							const vector<bool>& propagate_down, 
                                const vector<Blob<Dtype>*>& bottom);
        
        Blob<Dtype> spatial_sum_multiplier_;
    };
}
```

axpy_layer是Layer的派生类，头文件声明了必要的成员函数。

**axpy_layer.cpp**

```c++
#include "caffe/layers/axpy_layer.hpp"

namespace caffe{
	template <typename Dtype>
    void AxpyLayer<Dtype>::Reshape(const vectror<Blob<Dtype>*>& bottom,
                                  const vector<Blob<type>*>& top){
        CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)); //判断N是否相等
        CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1)); //判断C是否相等
        
        if(bottom[0]->num_axes()==4){
            CHECK_EQ(bottom[0]->shape(2), 1);
            CHECK_EQ(bottom[0]->shape(3), 1);
        }
        CHECK(bottom[1]->shape() == bottom[2]->shape()); //判断两个shape[N,C,H,W]的是否相等
        top[0]—>ReshapeLike(*bottom[1]);
        int spatial_dim = bottom[1]->count(2); //spatial_dim = H*W
        if(spatial_sum_multiplier_.count() < spatial_dim){
            spatial_sum_multiplier_.Reshape(vector<int>(1, spatial_dim));
            caffe_set(spatial_dim, Dtype(1), spatial_sum_multiplier_.mutable_cpu_data());
        }
    }
    
    template <typename Dtype>
    void AxpyLayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        int channel_dim = bottom[1]->channels(); // C
        int spatial_dim = bottom[1]->count(2); // H*W
        
        const Dtype* scable_data = bottom[0]->cpu_data(); 
        
        const Dtype* x_data = bottom[1]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_copy(bottom[2]->count, bottom[2]->cpu_data(), top_data);
        for(int n = 0; n < bottom[1]->num(), ++n){
            for(int c = 0; c < channel_dim; ++c){
                int scale_offset = n * channel_dim + c;
                caffe_axpy(spatial_dim, scale_data[scale_offset],
                          x_data + scale_offset * spatial_dim,
                          top_data + scale_offset * spatial_dim);
            }
        }
    }
    
    template <typename Dtype>
    void AxpyLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<type>*>& bottom){
        const int count = top[0]->count();
        const Dtype* top_diff = top[0]->cpu_diff();
        if(propagate_down[0]){
            int spatial_dim = bottom[1]->count(2); // H*W
            const Dtype* x_data = bottom[1]->cpu_data();
            Dtype* x_diff = bottom[1]->mutable_cpu_diff();
            Dtype* scale_diff = bottom[0]->mutable_cpu_diff();
            
            caffe_mul(count, top_diff, x_data, x_diff);
            caffe_set(bottom[0]->count(), Dtype(0), scale_diff);
            caffe_cpu_gemv(CblasNoTrans, bottom[0]->count(), spatial_dim, Dtype(1),
                          x_diff, spatial_sum_multiplier_.cpu_data(), Dtype(1), 		
                          scale_diff);
            if(!propagate_dowm[1]){
                caffe_set(bottom[1]->count(), Dtype(0), x_diff);
            }
        }
        
        if(propgate_down[0]){
            int channel_dim = bottom[1]->channels();
            int spatial_dim = bottom[1]->count(2);
            
            const Dtype* scale_data = bottom[0]->cpu_data();
            Dtype* x_diff = bottom[1]->mutable_cpu_diff();
            for(int n = 0; n < bottom[1]->num(); ++n){
                for(int c = 0; c < channel_dim; ++c){
                    int scale_offset = n * channel_dim + c;
                    caffe_cpu_scale(spatial_dim, scale_data[scale_offset],
                                   top_diff + scale_offset*spatial_dim,
                                   x_diff + scale_offset*spatial_dim);
                }
            }
        }
        
        if(propagate_down[2]){
            caffe_copy(count, top_diff, bottom[2]->mutable_cpu_diff());
        }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(AxpyLayer)
    #endif
    
    INSTANTIATE_CLASS(AxpyLayer);
    REGISTER_LAYER_CLASS(Axpy); 
}
```
1. 成员函数Reshape里面的caffe_set函数定义在math_functions.hpp中，如下所示：

   ```c++
   template <typename Dtype>
   void caffe_set(const int N, const Dtype alpha, Dtype *X);
   ```

   其实现在math_functions.cpp中，并进行实显式 实例化

   ```c++
   template <typename Dtype>
   void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
   	if (alpha == 0) {
   		memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
   		return;
   	}
   	for (int i = 0; i < N; ++i) {
   		Y[i] = alpha;
   	}
   }
   
   template void caffe_set<int>(const int N, const int alpha, int* Y);
   template void caffe_set<float>(const int N, const float alpha, float* Y);
   template void caffe_set<double>(const int N, const double alpha, double* Y);
   ```

   

2. 成员函数Forward_cpu里面的caffe_copy函数在math_functions.hpp中, 如下所示：

   ```C++
   template <typename Dtype>
   void caffe_copy(const int N, const Dtype *X, Dtype *Y);
   ```

   其实现在math_functions.cpp中，并进行实显式 实例化

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

3. 成员函数Forward_cpu里面的caffe_axpy函数在math_functions.hpp中, 如下所示：

   ```c++
   template <typename Dtype>
   void caffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
   ```

   其实现在math_functions.cpp中，并进行实显式 实例化

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

   cblas_saxpy和cblas_daxpy是OpenBlas的函数计算`Y[i] = alpha*X[i]+Y[i]`

   ```cc
   void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
   
   void cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
   ```

   其中incx、incy是间隔步长

4. 如何求导的？f(x,y) = a*x + y -->分别对a、x和y求导。

   成员函数Backward_cpu里面的caffe_cpu_gemv函数在math_functions.hpp中声明，如下所示：

   ```c++
   template <typename Dtype>
   void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
       const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
       Dtype* y);
   ```

   在math_functions.cpp中进行实例化，如下所示：

   ```c++
   template <>
   void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
       const int N, const float alpha, const float* A, const float* x,
       const float beta, float* y) {
     cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
   }
   
   template <>
   void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
       const int N, const double alpha, const double* A, const double* x,
       const double beta, double* y) {
     cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
   }
   ```

   cblas_sgemv和cblas_dgemv在OpenBlas定义如下，计算`y = alpha*A*x + beta*y`

   ```c++
   void cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
   
   void cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
   ```

   成员函数caffe_cpu_scale在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
   ```

   在math_function.cpp中进行实例化，如下所示：

   ```c++
   template <>
   void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                               float* y) {
     cblas_scopy(n, x, 1, y, 1);
     cblas_sscal(n, alpha, y, 1);
   }
   
   template <>
   void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                                double* y) {
     cblas_dcopy(n, x, 1, y, 1);
     cblas_dscal(n, alpha, y, 1);
   }
   ```

   cblas_scopy和cblas_dcopy在OpenBlas定义如下，复制一个向量到另外一个向量：

   ```c++
   void cblas_scopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
   
   void cblas_dcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
   ```

   cblas_sscal和cblas_dscal在OpenBlas定义如下，X[i] = alpha*X[i]：

   ```c++
   void cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, OPENBLAS_CONST blasint incX);
   
   void cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, double *X, OPENBLAS_CONST blasint incX);
   ```

   

**axpy_layer.cu**

```c++
#include "caffe/layers/axpy_layer.hpp"

namespace caffe{
    template <typename Dtype>
    __global__ void AxpyForward(const int count, const int spatial_dim, const Dtype* scale_data,
                               const Dtype* x_data, const Dtype* y_data, Dtype* out_data){
        CUDA_KERNEL_LOOP(index, count){
            out_data[index] = scale_data[index / spatial_dim] * x_data[index] + y_data[index];
        }
    }
    
    template <typename Dtype>
    void AxpyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
    {
        const Dtype* scale_data = bottom[0]->gpu_data();
        const Dtype* x_data = bottom[1]->gpu_data();
        const Dtype* y_data = bottom[2]->gpu_data();
        
        Dtype* out_data = top[0]->mutable_gpu_data();
        const int count = bottom[1]->count();
        
        AxpyForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
                            bottom[1]->count(2), scale_data, x_data, y_data, out_data);
    }
    
    template <typename Dtype>
    __global__ void AxpyBackwardScale(const int outer_dim, const int spatial_dim, const Dtype* x_data,
                                     const Dtype* top_diff, Dtype* scale_diff){
        __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
        unsigned int tid = threadIdx.x;
        buffer[tid] = 0;
        __syncthreads();
        
        for(int j = tid; j < spatial_dim; j +=  blockDim.x){
            int offset = blockIdx.x * spatial_dim + j;
            buffer[tid] += top_diff[offset] * x_data[offset];
        }
        __syncthreads();
        
        // 规约求和
        for(int i = blockDim.x / 2; i > 0; i >> 1){
            if(tid < i){ // 该判断条件很重要
                buffer[threadIdx.x] += buffer[threadIdx.x + i];
            }
            __syncthreads();
        }
        
        if(tid == 0){
            scale_diff[blockIdx.x] = buffer[0];
        }
    }
    
    template <typename Dtype>
    void AxpyLayer<Dtype>::AxpyBackwardX(const int count, const int spatial_dim, const Dtype* scale_data,
                                        const Dtype* top_diff, Dtype* out){
        CUDA_KERNEL_LOOP(index, count){
            out[index] = scale[index / spatial_dim] * top_diff[index];
        }
    }
    
    template <typename Dtype>
    void AxpyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
                            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
        const int count = top[0]->count();
        const Dtype* top_diff = top[0]->gpu_diff();
       	
        if(propagate_down[0]){
            int outer_num = bottom[1]->count(0,2);
            AxpyBackwardScale<Dtype><<<outer_num, CAFFE_CUDA_NUM_THREADS>>>(
            	outer_num, bottom[1]->count(2),
                bottom[1]->gpu_data(), top_diff,
                bottom[0]->mutable_gpu_diff());
        }
        
        if(propagate_down[1]){
            AxpyBackwardX<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            	count, top[0]->count(2),
            	bottom[0]->gpu_data(), top_diff, 
            	bottom[1]->mutable_gpu_diff());
        }
        
        if(propagate_down[2]){
            caffe_copy(count, top_diff, bottom[2]->mutable_gpu_diff());
        }
        CUDA_POST_KERNEL_CHECK;
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(AxpyLayer);
}
```

