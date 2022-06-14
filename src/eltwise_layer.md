# Eltwise Layer

**elementwise layer**在源文件**eltwise_layer.hpp、eltwise_layer.cpp、eltwise_layer.cu**中实现。

**eltwise_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

// Compute elementwise operations, such as product and sum, along multiple input Blobs.
namespace caffe{
    
	template <typename Dtype>
    class EltwiseLayer: public Layer<Dtype>{
    public:
        explicit EltwiseLayer(const LayerParameter& param)
            ：Layer<Dtype>(param){}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const {return "Eltwise";}
        virtual inline int MinBottomBlobs() const {return 2;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        
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
        
        EltwiseParameter_EltwiseOp op_;
        vector<Dtype> coeffs_;
        Blob<int> max_idx_;
        Blob<Dtype> sort_temp_;
        
        bool stable_prod_grad_;
    };
    
}
```

**eltwise_layer.cpp**

```c++
#include <cfloat>
#include <vector>

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{    
    
    template <typename Dtype>
    void EltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        CHECK(this->layer_param().eltwise_param().coeff_size() == 0
              || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
            "Eltwise Layer takes one coefficient per bottom blob.";
        CHECK(!(this->layer_param().eltwise_param().operation() == 
             EltwiseParameter_EltwiseOp_PROD && this->layer_param().eltwise_param().coeff_size())) <<
            "Eltwise layer only takes coefficients for summation.";
        op_ = this->layer_param_.eltwise_param().operation();
        coeffs_ = vector<Dtype>(bottom.size(), 1);
        
        if(this->layer_param().eltwise_param().coeff_size()){
            for(int i=0; i < bottom.size(); ++i){
                coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
            }
        }
        stable_prod_grad_ = this->layer_param_.eltwise_param().satble_prod_grad();
    }
    
    template <typename Dtype>
    void EltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
        for(int i=1; i<bottom.size(); ++i){
            CHECK(bottom[0]->shape() == bottom[i]->shape())
                << "bottom[0]: " << bottom[0]->shape_string()
                << ", bottom[" << i << "]:" << bottom[i]->shape_string();
        }
        
        top[0]->ReshapeLike(*bottom[0]);
        if(this->layer_param_.eltwise_param().operation() == 
           EltwiseParameter_EltwiseOp_MAX && top.size() == 1){
            max_idx_.Reshape(bottom[0]->shape());
        }
        
        if(this->layer_param_.eltwise_param().operation() == 
           EltwiseParameter_EltwiseOp_SORT && top.size() == 1){
            sort_tem_.Reshape(bottom[0]->shape());
        }
    }
    
    template <typename Dtype>
    void EltwiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top){
        int* mask = NULL;
        const Dtype* bottom_data_a = NULL;
        const Dtype* bottom_data_b = NULL;
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_cpu_data();
        
        switch(op_){
            case EltwiseParameter_EltwiseOp_PROD:
                caffe_mul(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
                for(int i=2; i<bottom.size(); ++i){
                    caffe_mul(count, top_data, bottom[i]->cpu_data(), top_data);
                }
                break;
            case EltwiseParameter_EltwiseOp_SUM:
                caffe_set(count, Dtype(0), top_data);
                for(int i = 0; i < bottom.size(); ++i){
                    caffe_axpy(count, coeffs_[i], bottom[i]->cpu_data(), top_data);
                }
                break;
            case EltwiseParameter_EltwiseOp_MAX:
                mask = max_idx_.mutable_cpu_data();
                caffe_set(count, -1, mask);
                caffe_set(count, Dtype(-FLT_MAX), top_data);
                
                bottom_data_a = bottom[0]->cpu_data();
                bottom_data_b = bottom[1]->cpu_data();
                
                for(int idx = 0; idx < count; ++idx){
                    if(bottom_data_a[idx] > bottom_data_b[idx]){
                        top_data[idx] = bottom_data_a[idx];
                       	mask[idx] = 0;
                    }
                    else{
                        top_data[idx] = bottom_data_data_b[idx];
                        mask[idx] = 1;
                    }
                }
                
                for(int blob_idx = 2; blob_idx < bottom.size(); ++blob_idx){
                    bottom_data_b = bottom[blob_idx]->cpu_data();
                    for(int idx = 0; idx < count; ++idx){
                        if(bottom_data_b[idx] > top_data[idx]){
                            top_data[idx] = bottom_data_b[idx];
                            mask[idx] = blob_idx;
                        }
                    }
                }
                break;
            case EltwiseParameter_EltwiseOp_SORT:
                caffe_set(count, Dtype(1), top_data);
                for(int i = 0; i < bottom.size(); ++i){
                    caffe_copy(count, bottom[i]->cpu_data(), sort_temp_.mutable_cpu_data());
                    caffe_add_scalar(count, Dtype(1), sort_temp_.mutable_cpu_data());
                    caffe_mul(count, top_data, sort_temp_.cpu_data(), top_data);
                }
                break;
            default:
                LOG(FATAL) << "Unknown elementwise operation.";
        }
    }
    
    template <typename Dtype>
    void EltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        const int* mask = NULL;
        const int count = top[0]->count();
        const Dtype* top_data = top[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        
        for(int i = 0; i < bottom.size(); ++i){
            if(propagate_down[i]){
                const Dtype* bottom_data = bottom[i]->cpu_data();
                Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
                switch(op_){
                    case EltwiseParameter_EltwiseOp_PROD:
                        if(stable_prod_grad_){
                            bool initialized = false;
                            for(int j = 0; j < bottom.size(); ++j){
                                if(i == j) {continue;}
                                if(!initialized){
                                    caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
                                    initialized = true;
                                }
                                else{
                                    caffe_mul(count, bottom[j]->cpu_data(), bottom_diff, bottom_diff);
                                }
                            }
                        }
                        else{
                            caffe_div(count, top_data, bottom_data, bottom_diff);
                        }
                        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
                        break;
                        if(stable_prod_grad_){
                            bool initialized = false;
                            for(int j = 0; j < bottom.size(); ++j){
                                if(i==j){continue;}
                                if(!initialized){
                                    caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
                                    caffe_add_scalar(count, Dtype(1), bottom_diff);
                                    initialized = true;
                                }
                                else{
                                    caffe_copy(count, bottom[j]->cpu_data(), sort_temp_.mutable_cpu_data());
                                    caffe_add_scalar(count, Dtype(1), sort_temp_.mutable_cpu_data());
                                    caffe_mul(count, sort_temp_.cpu_data(), bottom_diff, bottom_diff);
                                }
                            }
                        }
                        else{
                            caffe_copy(count, bottom_data, sort_temp_.mutable_cpu_data());
                            caffe_add_scalar(count, Dtype(1), sort_temp_.mutable_cpu_data());
                            caffe_div(count, top_data, sort_temp_.cpu_data(), bottom_diff);
                        }
                        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
                        break;
                    case EltwiseParameter_EltwiseOp_MAX:
                        mask = max_idx_.cpu_data();
                        for(int index = 0; index < count; ++index){
                            Dtype gradient = 0;
                            if(mask[index] == 1){
                                gradient += top_diff[index];
                            }
                            bottom_diff[index] = gradient;
                        }
                        break;
                    case EltwiseParameter_EltwiseOp_SORT:
                        if(stable_prod_grad_){
                            bool initialized = false;
                            for(int j = 0; j < bottom.size(); ++j){
                                if(i == j) {continue;}
                                if(!initialized){
                                    caffe_copy(count, bottom[j]->cpu_data(), bottom_diff);
                                    caffe_add_scalar(count, Dtype(1), bottom_diff);
                                    initialized = true;
                                }
                                else{
                                    caffe_copy(count, bottom[j]->cpu_data(), sort_temp_.mutable_cpu_data());
                                    caffe_add_scalar(count, Dtype(1), sort_temp_.mutable_cpu_data());
                                    caffe_mul(count, sort_temp_.cpu_data(), bottom_diff, bottom_diff);
                                }
                            }
                        }
                        else{
                            caffe_copy(count, bottom_data, sort_temp_.mutable_cpu_data());
                            caffe_add_scalar(count, Dtype(1), sort_temp_.mutable_cpu_data());
                            caffe_div(count, top_data, sort_temp_.cpu_data(), bottom_diff);
                        }
                        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
                        break;
                    default:
                        LOG(FATAL) << "Unknown elementwise operation.";
                }
            }
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(EltwiseLayer);
#endif
    
    INSTANTIATE_CLASS(EltwiseLayer);
    REGISTER_LAYER_CLASS(Eltwise);
}
```

 1. caffe_mul在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c
    template <>
    void caffe_mul<float>(const int n, const float* a, const float* b, float* y) {
      vsMul(n, a, b, y);
    }
    
    template <>
    void caffe_mul<double>(const int n, const double* a, const double* b, double* y) {
      vdMul(n, a, b, y);
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

 2. caffe_axpy在math_functions.hpp中声明如下：

    ```
    template <typename Dtype>
    void caffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) 
    { 
        cblas_saxpy(N, alpha, X, 1, Y, 1); 
    }
    
    template <>
    void caffe_axpy<double>(const int N, const double alpha, const double* X, double* Y) 
    { 
        cblas_daxpy(N, alpha, X, 1, Y, 1); 
    }
    ```

    ```c++
    void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
    
    void cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
    ```

 3. caffe_copy在math_functions.hpp中声明如下：

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

 5. caffe_cpu_scale在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

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

    ```c++
    void cblas_scopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, 
                     float *y, OPENBLAS_CONST blasint incy);
    void cblas_dcopy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, 
                     double *y, OPENBLAS_CONST blasint incy);
    
    void cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, 
                     OPENBLAS_CONST blasint incX);
    void cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, double *X, 
                     OPENBLAS_CONST blasint incX);
    ```

**eltwise_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/eltwise.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    __global__ void MaxForward(const int nthreads, const Dtype* bottom_data_a,
                              const Dtype* bottom_data_b, const int blob_idx, Dtype* top_data, int* mask){
        CUDA_KERNEL_LOOP(index, nthreads){
            Dtype maxval = -FLT_MAX;
            int maxidx = -1;
            if(bottom_data_a[index] > bottom_data_b[index]){
                if(blob_index == 0){
                    maxval = bottom_data_a[index];
                    top_data[index] = maxval;
                    maxidx = blob_idx;
                    mask[index] = maxidx;
                }
            }
            else{
                maxval = bottom_data_b[index];
                top_data[index] = maxval;
                maxidx = blob_idx + 1;
                mask[index] = maxidx;
            }
        }
    }
    
    template <typename Dtype>
    void EltwiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        int* mask = NULL;
        const int count = top[0]->count();
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        switch(op_){
            case EltwiseParameter_EltwiseOp_PROD:
                caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), top_data);
                for(int i = 2; i < bottom.size(); ++i){
                    caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
                }
                break;
            case EltwiseParameter_EltwiseOp_SUM:
                caffe_gpu_set(count, Dtype(0), top_data);
                for(int i = 0; i < bottom.size(); ++i){
                    caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
                }
                break;
            case EltwiseParameter_EltwiseOp_MAX:
                mask = max_idx_.mutable_gpu_data();
                MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                	count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
                
                for(int i = 2; i < bottom.size(); ++i){
                    MaxForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                    	count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
                }
                break;
            default:
                LOG(FATAL) << "Unknown elementwise operation.";
        }
    }
    
    template <typename Dtype>
    __gloabl__ void MaxBackward(const int nthreads, const Dtype* top_diff, const int blob_idx,
                               const int* mask, Dtype* bottom_diff){
        CUDA_KERNEL_LOOP(index, nthreads){
            Dtype gradient = 0;
            if(mask[index] == blob_idx){
                gradient += top_diff[index];
            }
            bottom_diff[index] = gradient;
        }
    }
    
    template <typename Dtype>
    void EltwiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        const int* mask = NULL;
        const int count = top[0]->count();
        const Dtype* top_data = top[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        
        for(int i = 0; i < bottom.size(); ++i){
            if(propagate_down[i]){
                const Dtype* bottom_data = bottom[i]->gpu_data();
                Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
                
                switch(op_){
                    case EltwiseParameter_EltwiseOp_PROD:
                        if(stable_prod_grad_){
                            bool initialized = false;
                            for(int j = 0; j < bottom.size(); ++j){
                                if(i == j) {continue;}
                                if(!initialized){
                                    caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
                                    initialized = true;
                                }
                                else{
                                    caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff, bottom_diff);
                                }
                            }
                        }
                        else{
                            caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
                        }
                        caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
                        break;
                    case EltwiseParameter_EltwiseOp_SUM:
						if(coeffs_[i] == Dtype(1.)){
                            caffe_copy(count, top_diff, bottom_diff);
                        }
                        else{
                            caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
                        }
                        break;
                    case EltwiseParameter_EltwiseOp_MAX:
                        mask = max_idx_.gpu_data();
                        MaxBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                        	count, top_diff, i, mask, bottom_diff);
                        break;
                    default:
                        LOG(FATAL) << "Unknown elementwise operation."; 
                }
            }
        }
    }
        
    INSTANTIATE_LAYER_GPU_FUNCS(EltwiseLayer);
}
```

 1. caffe_gpu_mul在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void mul_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] * b[index];
      }
    }
    
    template <>
    void caffe_gpu_mul<float>(const int N, const float* a, const float* b, float* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
    }
    
    template <>
    void caffe_gpu_mul<double>(const int N, const double* a, const double* b, double* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
    }
    ```

 2. caffe_gpu_axpy在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
    ```

    在math_functions.cu中进行实例化

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

 3. caffe_gpu_div在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_fnctions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void div_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] / b[index];
      }
    }
    
    template <>
    void caffe_gpu_div<float>(const int N, const float* a, const float* b, float* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, b, y);
    }
    
    template <>
    void caffe_gpu_div<double>(const int N, const double* a, const double* b, double* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, a, b, y);
    }
    ```

 4. caffe_gpu_scale在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
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

