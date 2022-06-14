# Power Layer
power layer在power_layer.hpp, power_layer.cpp和power_layer.cu中实现。power layer计算公式：
$$
y = (\alpha x+\beta)^{\gamma}
$$
其中，

$$
\gamma: power \\
\alpha: scale   \\
\beta: shift    \\
$$
power layer求导：
$$
y^{'} = \alpha \gamma (\alpha x+\beta)^{\gamma-1}
$$
**power_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"


namespace caffe{

   template <typename Dtype>
   class PowerLayer: public NeuronLayer<Dtype>{
   public:
       explicit PowerLayer(const LayerParameter& param)
           : NeuronLayer<Dtype>(param){}
       
       virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
       virtual inline const char* type() const {return "Power";}
       
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
       
       Dtype power_;
       Dtype scale_;
       Dtype shift_;
       Dtype diff_scale_;
   };
}
```

**power_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
   
    template <typename Dtype>
    void PowerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        NeuronLayer<Dtype>::LayerSetUp(bottom, top);
        power_ = this->layer_param_.power_param().power();
        scale_ = this->layer_param_.power_param().scale();
        shift_ = this->layer_param_.power_param().shift();
        diff_scale_ = power_ * scale_;
    }
    
    template <typename Dtype>
    void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        
        // 如果diff_scale为零，那么power_=0或者scale_=0
        if(diff_scale_ == Dtyep(0)){
            // 若power_=0, value=1
            // 若scale_=0, value=gamma^power
            Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
            caffe_set(count, value, top_data);
            return;
        }
        
        // 
        const Dtype* bottom_data = bottom[0]->cpu_data();
        caffe_copy(count, bottom_data, top_data);
        
        if(scale_ != Dtype(1)){
            caffe_scale(count, scale_, top_data);
        }
        
        if(shift_ != Dtype(0)){
            caffe_add_scalar(count, shift_, top_data);
        }
        
        if(power_ != Dtype(1)){
            caffe_powx(count, top_data, power_, top_data);
        }
    }
    
    template <typename Dtype>
    void PowerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            const Dtype* top_diff = top[0]->cpu_diff();
            
            // 如果 alpha*gamma=0或者power=1, 那么bottom_diff=0
            if(diff_scale_ == Dtype(0) || power_ == Dtype(1)){
                caffe_set(count, diff_scale, bottom_diff);
            }
            else
            {
                // y' = scale*power*(shift+scale*x)^(power-1)
                // 	  = diff_scale*y/(shift+scale*x)
                const Dtype* bottom_data = bottom[0]->cpu_data();
                if(power_ == Dtype(2)){
                    // Special case for y=(shift+scale*x)^2
                    // --> y'=2*scale*(shift+scale*x)
                    //       =diff_scale*shift+diff_scale*scale*x
                    
                    // 计算diff_scale_*scale_*x
                    caffe_cpu_axpby(count, diff_scale_ * scale_, bottom_data, Dtype(0), bottom_diff);
                    // 如果shift!=0, 那么diff_scale*shift=0
                    if(shift_ != Dtype(0)){
                        caffe_add_scalar(count, diff_scale_ * shift_, bottom_diff);
                    }
                }
                else if(shift_ == Dtype(0)){
                    // Special case for y=scale*power*(scale*x)^power
                    // --> y'=scale*power*(scale*x)^(power-1)
                    //       =scale*power*(scale*x)^power*(scale*x)^(-1)
                    //       =power*(scale*x)^power*(x)^-1
                    //       =power*y/x
                    const Dtype* top_data = top[0]->cpu_data();
                    caffe_div(count, top_data, bottom_data, bottom_diff);
                    caffe_scal(count, power_, bottom_diff);
                }
                else
                {
                    caffe_copy(count, bottom_data, bottom_diff);
                    if(scale_ != Dtype(1)){
                        caffe_scal(count, scale_, bottom_diff);
                    }
                    if(shift_ != Dtype(0)){
                        caffe_add_scalar(count, shift_, bottom_diff);
                    }
                    const Dtype* top_data = top[0]->cpu_data();
                    // 计算(scale*x+shift)^(power-1)
                    caffe_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
                    if(diff_scale_ != Dtype(1)){
                        // 计算scale*power*(scale*x+shift)^(power-1)
                        caffe_scal(count, diff_scale_, bottom_diff);
                    }
                }
            }
            if(diff_scale_ != Dtype(0)){
                caffe_mul(count, top_diff, bottom_diff, bottom_diff);
            }
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(PowerLayer);
#endif
    
    INSTANTIATE_CLASS(PowerLayer);
    REGISTER_LAYER_CLASS(Power);
}

```

 1. caffe_set在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_set(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```C++
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

 2. caffe_copy在math_functions.hpp声明如下：

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
        } else {
          memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
        }
      }
    }
    
    template void caffe_copy<bool>(const int N, const bool* X, bool* Y);
    template void caffe_copy<int>(const int N, const int* X, int* Y);
    template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
        unsigned int* Y);
    template void caffe_copy<float>(const int N, const float* X, float* Y);
    template void caffe_copy<double>(const int N, const double* X, double* Y);
    ```

 3. caffe_scal在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_scal(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

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

 4. caffe_add_scalar在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_add_scalar(const int N, const float alpha, float* Y) {
      for (int i = 0; i < N; ++i) {
        Y[i] += alpha;
      }
    }
    
    template <>
    void caffe_add_scalar(const int N, const double alpha, double* Y) {
      for (int i = 0; i < N; ++i) {
        Y[i] += alpha;
      }
    }
    ```

 5. caffe_powx在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_powx<float>(const int n, const float* a, const float b,
        float* y) {
      vsPowx(n, a, b, y);
    }
    
    template <>
    void caffe_powx<double>(const int n, const double* a, const double b,
        double* y) {
      vdPowx(n, a, b, y);
    }
    ```

    ```c++
    #define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
      template<typename Dtype> \
      void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
        CHECK_GT(n, 0); CHECK(a); CHECK(y); \
        for (int i = 0; i < n; ++i) { operation; } \
      } \
      inline void vs##name( \
        const int n, const float* a, const float b, float* y) { \
        v##name<float>(n, a, b, y); \
      } \
      inline void vd##name( \
          const int n, const double* a, const float b, double* y) { \
        v##name<double>(n, a, b, y); \
      }
    
    DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b));
    ```

 6. caffe_div在math_functions.hpp中声明：

    ```c++
    template <typename Dtype>
    void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_div<float>(const int n, const float* a, const float* b,
        float* y) {
      vsDiv(n, a, b, y);
    }
    
    template <>
    void caffe_div<double>(const int n, const double* a, const double* b,
        double* y) {
      vdDiv(n, a, b, y);
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
    
    DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
    DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
    DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
    DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
    ```

 7. caffe_mul在math_functions.hpp中声明如下:

    ```c++
    template <typename Dtype>
    void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
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

**power_layer.cu**

```c++
#include <vector>

#include "caffe/layers/power_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    void PowerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();
        
        if(diff_scale_ == Dtype(0)){
            Dtype value = (power_ == 0) ? Dtyep(1) : pow(shift_, power_);
            caffe_gpu_set(count, value, top_data);
            return;
        }
        
        const Dtype* bottom_data = bottom[0]->gpu_data();
        caffe_copy(count, bottom_data, top_data);
        
        if(scale_ != Dtype(1)){
            caffe_gpu_scal(count, scale_, top_data);
        }
        if(shift_ != Dtype(0)){
            caffe_gpu_add_scalar(count, shift_, top_data);
        }
        if(power_ != Dtype(1)){
            caffe_gpu_powx(count, top_data, power_, top_data);
        }
    }
    
    template <typename Dtype>
    void PowerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            const Dtype* top_diff = top[0]->gpu_diff();
            
            if(diff_scale_ == Dtype(0) || power_ == Dtype(1)){
                caffe_gpu_set(count, diff_scale_, bottom_diff);
            }
            else{
                const Dtype* bottom_data = bottom[0]->gpu_data();
                if(power_ == Dtyep(2)){
                    caffe_gpu_axpby(count, diff_scale_ * scale_, bottom_data, Dtype(0), bottom_diff);
                    if(shift_ != Dtyep(0)){
                        caffe_gpu_add_scalar(count, diff_scale_ * shift_, bottom_diff);
                    }
                }
                else if(shift_ == Dtype(0))
                {
                    const Dtype* top_data = top[0]->gpu_data();
                    caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
                    caffe_gpu_scal(count, power_, bottom_diff);
                } 
                else
                {
                    caffe_copy(count, bottom_data, bottom_diff);
                    if(scale_ != Dtype(1)){
                        caffe_gpu_scal(count, scale_, bottom_diff);
                    }
                    if(shift_ != Dtype(0)){
                        caffe_gpu_add_scalar(count, shift_, bottom_diff);
                    }
                    const Dtype* top_data = top[0]->gpu_data();
                    caffe_gpu_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
                    if(diff_scale_ != Dtype(1)){
                        caffe_gpu_scal(count, diff_scale_, bottom_diff);
                    }
                }
            }
            caffe_gpu_mul(count, top_diff, bottom_diff, bottom_diff);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(PowerLayer);
}
```

 1. caffe_gpu_set在math_function.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```C++
    template <typename Dtype>
    __global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = alpha;
      }
    }
    
    template <typename Dtype>
    void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
      if (alpha == 0) {
        CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
        return;
      }
      // NOLINT_NEXT_LINE(whitespace/operators)
      set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, alpha, Y);
    }
    
    template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
    template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
    template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);
    ```

 2. caffe_gpu_scal在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cu中进行实例化

    ```c++
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

 3. caffe_gpu_add_scalar在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cu中进行实例化

    ```C++
    template <typename Dtype>
    __global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] += alpha;
      }
    }
    
    template <>
    void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, alpha, Y);
    }
    
    template <>
    void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, alpha, Y);
    }
    ```

 4. caffe_gpu_powx在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void powx_kernel(const int n, const Dtype* a,
        const Dtype alpha, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = pow(a[index], alpha);
      }
    }
    
    template <>
    void caffe_gpu_powx<float>(const int N, const float* a,
        const float alpha, float* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, alpha, y);
    }
    
    template <>
    void caffe_gpu_powx<double>(const int N, const double* a,
        const double alpha, double* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, alpha, y);
    }
    ```

 5. caffe_gpu_axpby在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
        const Dtype beta, Dtype* Y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <>
    void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
        float* Y) {
      CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
    }
    
    template <>
    void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
        double* Y) {
      CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
    }
    
    template <>
    void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
      CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    }
    
    template <>
    void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
      CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    }
    
    template <>
    void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
        const float beta, float* Y) {
      caffe_gpu_scal<float>(N, beta, Y);
      caffe_gpu_axpy<float>(N, alpha, X, Y);
    }
    
    template <>
    void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
        const double beta, double* Y) {
      caffe_gpu_scal<double>(N, beta, Y);
      caffe_gpu_axpy<double>(N, alpha, X, Y);
    }
    ```

 6. caffe_gpu_div在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void div_kernel(const int n, const Dtype* a,
        const Dtype* b, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] / b[index];
      }
    }
    
    template <>
    void caffe_gpu_div<float>(const int N, const float* a,
        const float* b, float* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, b, y);
    }
    
    template <>
    void caffe_gpu_div<double>(const int N, const double* a,
        const double* b, double* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, b, y);
    }
    ```

 7. caffe_gpu_mul在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

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
      mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, b, y);
    }
    
    template <>
    void caffe_gpu_mul<double>(const int N, const double* a,
        const double* b, double* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, b, y);
    }
    ```

​		

​		
