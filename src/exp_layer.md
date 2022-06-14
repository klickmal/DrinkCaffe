# Exp Layer
Exp Layer计算公式：
$$
f = \gamma^{\alpha*x+\beta} \\
$$
其中 
$$
\gamma: base \\
\alpha: scale   \\
\beta: shift    \\
$$

Exp Layer求导：
$$
f^{-1} = \alpha\ln\gamma*\gamma^{\alpha*x+\beta}
$$
exp_layer在源文件exp_layer.hpp, exp_layer.cpp和exp_layer.cu中实现。

**exp_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    template <typename>
    class ExpLayer: public NeuronLayer<Dtype>{
        public:
            explicit ExpLayer(const LayerParameter& param):
                NeuronLayer<Dtype>(param){}

            virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);

            virtual inline const char* type() const{return "Exp";}

        protected:
            virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);
            virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top);

            virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype*>& bottom);
            virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom);

            Dtype inner_scale_, outer_scale_;
        };
}
```

**exp_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void ExpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
    {
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);
        
       	const Dtype base = this->layer_param_.exp_param().base();
        if(base != Dtype(-1)){
            CHECK_GT(base, 0) << "base must be strictly positive.";
        }
        // If base == -1, interpret the base as e and set log_base = 1 exactly.
		// Otherwise, calculate its log explicitly.
        const Dtype log_base = (base == Dtype(-1)) ? Dtype(-1) : log(base);
        CHECK(!isnan(log_base))
            << "NaN result: log(base) = log(" << base << ")=" << log_base;
        CHECK(!isinf(log_base))
            << "Inf result: log(base) = log(" << base << ")=" << log_base;
        
        const Dtype input_scale = this->layer_param_.exp_param().scale();
        const Dtype input_shift = this->layer_param_.exp_param().shift();
        
        // inner_sclae_ = alpha * ln(gamma)
        inner_scale_ = log_base * input_scale;
        outer_scale_ = (input_shift == Dtype(0)) ? Dtype(1) :
        	((base != Dtype(-1))) ? pow(base, input_shift) : exp(input_shift));
    }
    
    template <typename Dtype>
    void ExpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
            const int count = bottom[0]->count();
        	const Dtype* bottom_data = bottom[0]->cpu_data();
        	Dtype* top_data = top[0]->mutable_cpu_data();
        	
        	// 换底公式： r^x = e^(x*lnr)
        	if(inner_scale_ == Dtype(1)){
                caffe_exp(count, bottom_data, top_data);
            }
        	else{
                caffe_cpu_scale(count, inner_scale_, bottom_data, top_data);
                caffe_exp(count, top_data, top_data);
            }
           if(outer_scale_ != Dtype(1)){
               caffe_scale(count, outer_scale_, top_data);
           }
        }
	
    template <typename Dtype>
    void ExpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& parapagate_down,
                                  const vector<Blob<Dtype>*>& bottom){
            if(propagate_down[0]){
                const int count = bottom[0]->count();
                const Dtype* top_data = top[0]->cpu_data();   
               	const Dtype* top_diff = top[0]->cpu_diff();
                Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
                
                caffe_mul(count, top_data, top_diff, bottom_diff);
                if(inner_scale_ != Dtype(1)){
                    caffe_scale(count, inner_scale_, bottom_diff);
                }
            }
        }

    #ifdef CPU_ONLY
        STUB_GPU(ExpLayer);
    #endif
    
        INSTANTIATE_CLASS(ExpLayer);
        REGISTER_LAYER_CLASS(Exp);
}

```

 1. 其中caffe_exp函数在math_functions.hpp中声明：

    ```c++
    template <typename Dtype>
    void caffe_exp(const int n, const Dtype* a, Dtype* y);
    ```

    caffe_exp在math_functions.cpp中实现：

    ```c++
    template <>
    void caffe_exp<float>(const int n, const float* a, float* y) {
    	vsExp(n, a, y);
    }
    
    template <>
    void caffe_exp<double>(const int n, const double* a, double* y) {
    	vdExp(n, a, y);
    }
    ```

    其中vsExp, vdExp定义如下：

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
    
    DEFINE_VSL_UNARY_FUNC(Sqr, y[i] = a[i] * a[i])
    DEFINE_VSL_UNARY_FUNC(Sqrt, y[i] = sqrt(a[i]))
    DEFINE_VSL_UNARY_FUNC(Exp, y[i] = exp(a[i]))
    DEFINE_VSL_UNARY_FUNC(Ln, y[i] = log(a[i]))
    DEFINE_VSL_UNARY_FUNC(Abs, y[i] = fabs(a[i]))
    ```

 2. caffe_cpu_scale在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
    ```

    caffe_cpu_scale在math_functions.cpp定义实现如下：

    ```c++
    template <>
    void caffe_cpu_scale<float>(const int n, const float alpha, const float *x, float* y) {
    	cblas_scopy(n, x, 1, y, 1);
    	cblas_sscal(n, alpha, y, 1);
    }
    
    template <>
    void caffe_cpu_scale<double>(const int n, const double alpha, const double *x, double* y) {
    	cblas_dcopy(n, x, 1, y, 1);
    	cblas_dscal(n, alpha, y, 1);
    }
    ```
    
    cblas_scopy、cblas_dcopy在OpenBLAS接口定义如下：
    
    ```c++
    void cblas_scopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, 
                     OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
    void cblas_dcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, 
                     OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
    ```
    
    cblas_sscal、cblas_dscal在OpenBLAS接口定义如下：
    
    ```c++
    void cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, 
                     float *X, OPENBLAS_CONST blasint incX);
    void cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, 
                     double *X, OPENBLAS_CONST blasint incX);
    ```
    
 3. caffe_scale在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_scal(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp实现如下：

    ```c++
    template <>
    void caffe_scal<float>(const int N, const float alpha, float *X) {
    	cblas_sscal(N, alpha, X, 1);
    }
    
    template <>
    void caffe_scal<double>(const int N, const double alpha, double *X) {
    	cblas_dscal(N, alpha, X, 1);
    }
    ```

 4. caffe_mul在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_mul<float>(const int n, const float* a, const float* b, float* y) {
    	vsMul(n, a, b, y);
    }
    
    template <> 
    void caffe_mul<double>(const int n, const double* a, const double* b, double* y) {
    	vdMul(n, a, b, y);
    }
    ```

    在mkl_alternate.hpp中定义宏函数

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
    
    DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
    DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
    DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
    DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])
    ```

**exp_layer.cu**

```c++
#include <vector>

#include "caffe/layers/exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namesapce caffe{
    
    template <typename Dtype>
    void ExpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
       const int count = bottom[0]->count();
       const Dtype* bottom_data = bottom[0]->gpu_data();
       Dtype* top_data = top[0]->mutable_gpu_data();
      	
       if(inner_scale_ == Dtype(1)){
           caffe_gpu_exp(count, bottom_data, top_data);
       }
        else{
            caffe_gpu_scale(count, inner_scale_, bottom_data, top_data);
            caffe_gpu_exp(count, top_data, top_data);
        }
        if(outer_scale_ != Dtype(1)){
            caffe_gpu_scal(count, outer_scale_, top_data);
        }
    }
    
    template <typename Dtype>
    void ExpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const int count = bottom[0]->count();
            const Dtype* top_data = top[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            
            caffe_gpu_mul(count, top_data, top_diff, bottom_diff);
            if(inner_scale_ != Dtype(1)){
                caffe_gpu_scal(count, inner_scale_, bottom_diff);
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(ExpLayer);
}
```

1. caffe_gpu_exp在math_functions.hpp声明如下：

   ```cc
   template <typename Dtype>
   void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);
   ```

   caffe_gpu_exp在math_functions.cu定义如下：

   ```c++
   template <typename Dtype>
   __global__ void exp_kernel(const int n, const Dtype* a, Dtype* y){
   	CUDA_KERNEL_LOOP(index, n){
           y[index] = exp(a[index]);
       }
   }
   
   template <>
   void caffe_gpu_exp<float>(const int N, const float* a, float* y){
       exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
   }
   
   template <>
   void caffe_gpu_exp<double>(const int N, const double* a, double* y){
       exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
   }
   ```

   caffe_gpu_scale在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
   ```

   caffe_gpu_scale在math_functions.cu定义如下：

   ```
   template <>
   void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
   		float* y) {
   	CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
   	CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
   }
   
   template <>
   void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
   		double* y) {
   	CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
   	CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
   }
   ```

2. caffe_gpu_mul在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
   ```

   在math_funtions.cu实现如下：

   ```cc
   template <typename Dtype>
   __global__ void mul_kernbel(const int N, const Dtype* a, const Dtype* b, Dtype* y){
       CUDA_KERNEL_LOOP(index, n){
           y[index] = a[index] * b[index];
       }
   }
   
   template <>
   void caffe_gpu_mul<float>(const int N, const float* a, const float* b, float* y){
       mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b , y);
   }
   
   template<>
   void caffe_gpu_mul<double>(const int N, const double* a, const double* b, double* y){
       mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
   }
   ```

3. caffe_gpu_scal在math_functions.hpp声明如下：

   ```c++
   template <typename Dtype>
   void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);
   ```

   在math_functions.cu进行实例化

   ```c++
   template <>
   void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
   	CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
   }
   
   template <>
   void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
   	CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
   }
   ```

   
