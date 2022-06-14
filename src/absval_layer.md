# Absval Layer

absolute layer在源文件**absval_layer.hpp、absval_layer.cpp**中实现。

**absval_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namesapce caffe{
	template <typename Dtype>
    class AbsValLayer: public NeuronLayer<Dtype>{
    public: 
      	explicit AbsValLayer(const LayerParameter& param):
        	NeuronLayer<Dtype>(param){}
      	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                              const vector<Blob<Dtype>*>& top);
      	virtual inline const char* type() const {return "AbsVal";}
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
    };
}
```

AbsValLayer是NeuronLayer的派生类，头文件声明了必要的成员函数。

**absval_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template <typename Dtype>
    void AbsValLayer<Dtype>::LayerSetUp(const vectror<Blob<Dtype>*>& bottom,
                                       const vector<Blob<type>*>& top){
        NeuronLayer<Dtype>::LayerSetUp(bottom, top);
        CHECK_NE(top[0], bottom[0]) << this->type() << "Layer does not" 
            "allow in-place computation.";
    }
    
    template <typename Dtype>
    void AbsValLayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_cpu_data();
        caffe_abs(count, bottom[0]->cpu_data(), top_data);
    }
    
    template <typename Dtype>
    void AbsValLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<type>*>& bottom){
        const int count = top[0]->count();
        const Dtype* top_diff = top[0]->cpu_diff();
        if(propagate_down[0]){
            const Dtype* bottom_data = bototm[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            caffe_cpu_sign(count, bottom_data, bottom_diff);
            caffe_mul(count, bottom_diff, top_diff, bottom_diff);
        }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(AbsValLayer)
    #endif
    
    INSTANTIATE_CLASS(AbsValLayer);
    REGISTER_LAYER_CLASS(AbsVal); 
}
```
1. 成员函数Forward_cpu()里面的caffe_abs()函数定义在math_functions.hpp中，如下所示：

   ```c++
   template <typename Dtype>
   void caffe_abs(const int n, const Dtype* a, Dtype* y);
   ```

   其实现在math_functions.cpp中，

   ```c++
   template <>
   void caffe_abs<float>(const int n, const float* a, float* y) {
       vsAbs(n, a, y);
   }
   
   template <>
   void caffe_abs<double>(const int n, const double* a, double* y) {
       vdAbs(n, a, y);
   }
   ```

   其中 vsAbs()、vdAbs()是定义的宏函数，

   ```c++
   #define DEFINE_VSL_UNARY_FUNC(name, operation) \
     template<typename Dtype> \
     void v##name(const int n, const Dtype* a, Dtype* y) { \
       CHECK_GT(n, 0); CHECK(a); CHECK(y); \
       for (int i = 0; i < n; ++i) { operation; } \
     } \
     inline void vs##name( \
       const int n, const float* a, float* y) { \
       v##name<float>(n, a, y); \
     } \
     inline void vd##name( \
         const int n, const double* a, double* y) { \
       v##name<double>(n, a, y); \
     }
   
   DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
   DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]));
   DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]));
   DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]));
   ```

   

2. 成员函数Backward_cpu()里面的caffe_cpu_sign()函数定义在math_functions.hpp中, 如下所示：

   ```C++
   #define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
     template<typename Dtype> \
     void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
       CHECK_GT(n, 0); CHECK(x); CHECK(y); \
       for (int i = 0; i < n; ++i) { \
         operation; \
       } \
     }
   
   // output is 1 for the positives, 0 for zero, and -1 for the negatives
   DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));
   ```

   caffe_sign()在math_functions.hpp定义如下：

   ```c++
   template<typename Dtype>
   inline int8_t caffe_sign(Dtype val) {
     return (Dtype(0) < val) - (val < Dtype(0));
   }
   ```
   
   caffe_mul()在math_functions.hpp声明如下：
   
   ```c++
   template <typename Dtype>
   void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
   ```
   
   ```C++
   template <>
   void caffe_mul<float>(const int n, const float* a, const float* b,
       float* y) {
     vsMul(n, a, b, y);
   }
   
   template <>
   void caffe_mul<double>(const int n, const double* a, const double* b,
       double* y) {
     vdMul(n, a, b, y);
   }
   ```
   
   ```c++
   // A simple way to define the vsl binary functions. The operation should
   // be in the form e.g. y[i] = a[i] + b[i]
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
   
   DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
   DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
   DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
   DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
   ```
   
   

**absval_layer.cu**

```c++
#include <vector>

#include "caffe/layers/absval_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    void AbsValLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        caffe_gpu_abs(count, bottom[0]->gpu_data(), top_data);
    }
    
    template <typename Dtype>
    void AbsValLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom){
        const int count = top[0]->count();
        const Dtype* top_diff = top[0]->gpu_diff();
        if(propagate_down[0]){
            const Dtype* bottom_data = bottom[0]->gpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            caffe_gpu_sign(count, bottom_data, bottom_diff);
            caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(AbsValLayer);
}
```

1. 成员函数Forward_gpu()中的caffe_gpu_abs()在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);
   ```

   在math_functions.cu实现如下：

   ```c++
   template <typename Dtype>
   __global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
     CUDA_KERNEL_LOOP(index, n) {
       y[index] = abs(a[index]);
     }
   }
   
   template <>
   void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
     // NOLINT_NEXT_LINE(whitespace/operators)
     abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
         N, a, y);
   }
   
   template <>
   void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
     // NOLINT_NEXT_LINE(whitespace/operators)
     abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
         N, a, y);
   }
   ```

   其中

   ```c++
   // CUDA: use 512 threads per block
   const int CAFFE_CUDA_NUM_THREADS = 512;
   
   // CUDA: number of blocks for threads.
   inline int CAFFE_GET_BLOCKS(const int N) {
     return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
   }
   ```

   ```C++
   // CUDA: grid stride looping
   #define CUDA_KERNEL_LOOP(i, n) \
     for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
          i < (n); \
          i += blockDim.x * gridDim.x)
   ```

2. 成员函数Backward_gpu()中的caffe_gpu_sign()在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void (const int n, const Dtype* x, Dtype* y);
   ```

   同时在math_functions.hpp定义宏函数如下：

   ```c
   #define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
   template<typename Dtype> \
   __global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
     CUDA_KERNEL_LOOP(index, n) { \
       operation; \
     } \
   } \
   template <> \
   void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
     /* NOLINT_NEXT_LINE(whitespace/operators) */ \
     name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
         n, x, y); \
   } \
   template <> \
   void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
     /* NOLINT_NEXT_LINE(whitespace/operators) */ \
     name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
         n, x, y); \
   }
   ```

   在math_functions.cu实现如下：

   ```c++
   DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
   		- (x[index] < Dtype(0)));
   ```

3. ```c++
   template <typename Dtype>
   void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
   ```

   在math_functions.cu中实现如下：

   ```c++
   template <typename Dtype>
   __global__ void mul_kernel(const int n, const Dtype* a,
   	const Dtype* b, Dtype* y) {
   	CUDA_KERNEL_LOOP(index, n) {
   			y[index] = a[index] * b[index];
   	}
   }
   
   template <>
   void caffe_gpu_mul<float>(const int N, const float* a,
   	const float* b, float* y) {
   		// NOLINT_NEXT_LINE(whitespace/operators)
   		mul_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (
   			N, a, b, y);
   	}
   
   template <>
   void caffe_gpu_mul<double>(const int N, const double* a,
   	const double* b, double* y) {
   	// NOLINT_NEXT_LINE(whitespace/operators)
   	mul_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (
   		N, a, b, y);
   }
   ```

   
