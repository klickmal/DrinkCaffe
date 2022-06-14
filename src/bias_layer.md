# Bias Layer

Bias layer功能：

* Computes a sum of two input Blobs, with the shape of the latter Blob

* "broadcast" to match the shape of the former. Equivalent to tiling

 * the latter Blob, then computing the elementwise sum.

   

 * The second input may be omitted, in which case it's learned as a parameter

 * of the layer. Note: in case bias and scaling are desired, both operations can

 * be handled by `ScaleLayer` configured with `bias_term: true`.



在源文件**bias_layer.hpp、bias_layer.cpp、bias_layer.cu**中实现。

**bias_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

    template <typename Dtype>
    class BiasLayer: public Layer<Dtype>{
    public:
        explicit BiasLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Bias"};
        virtual inline int MinBottomBlobs() const {return 1;}
        virtual inline int MaxBottomBlobs() const {return 2;}
        virtual inline int ExactNumTopBlobs() const {retun 1;}
        
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
    
    private:
        Blob<Dtype> bias_multiplier_;
        int outer_dim_, bias_dim_, inner_dim_, dim_;
    }
}
```



**bias_layer.cpp**

```c++
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{    
   
    template <typename Dtype>
    void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                     const vector<Blob<Dtype>*>& top){
        
        if(bottom.size() == 1 && this->blobs_.size() > 0){
            LOG(INFO) << "Skipping parameter initialization";
        }
        
        else if(bottom.size() == 1){
            // bias是可学习参数，对其初始化
            const BiasParameter& param = this->layer_param_.bias_param();
            const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
            
            const int num_axes = param.num_axis();
            
            // num_axes是大于等于-1的值
            CHECK_GE(num_axes, -1) << "num_axes must be non_negative, "
                << "or -1 to extend to the end of bottom[0]";
            
            if(num_axes >= 0){
                CHECK_GE(bottom[0]->num_axes, axis+num_axis)
                    << "bias blob's shape extends past bottom[0]'s shape when applied"
                    << "starting with bottom[0] axis = " << axis;
            }
            
            this->blobs_.resize(1);
            const vector<int>::const_iterator& shape_start = 
                bottom[0]->shape().begin() + axis;
            const vector<int>::const_iterator& shape_end = (num_axes == -1) ?
                bottom[0]->shape().end() : (shape_start + num_axes);
            vecrtor<int> bias_shape(shape_start, shape_end);
            this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
            share_ptr<Filler<Dtype>> filler(GetFiller<Dtype>(param.filler()));
            filler->Fill(this->blobs_[0].get());
        }
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }
    
    template <typename Dtype>
    void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top){
        const BiasParameter& param = this->layer_param_.bias_param();
        
        Blob<Dtype>* bias = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
        
        const int axis = (bias->num_axes() == 0) ? 0 : bottom[0]->CanonaicalAxisIndex(param.axis());
        
        CHECK_GE(bottom[0]->num_axes(), axis+bias->num_axes())
            << "bias blob's shape extends past bottom[0]'s shape when applied "
            << "starting with bottom[0] axes = " << axis;
        
        // 保证bias的shape和其从axis开始覆盖的bottom[0]的形状相同
        for(int i=0; i<bias->num_axes(); ++i){
            CHECK_EQ(bottom[0]->shape(axis + i), bias->shape(i))
                << "dimensiojn mismatch between bottom[0]->shape(" << axis + i
                <<  ") and bias->shape(" << i << ")";
        }
        
        // 假设bottom[0]的形状是[n,c,h,w], axis=1, num_axes=2
        // outer_dim_ = n
        // bias_dim_ = c*h
        // inner_dim_ = w
        // dim_ = c*h*w
        outer_dim_ = bottom[0]->count(0, axis);
        bias_dim_ = bias_->count();
        inner_dim_ = bottom[0]->count(axis + bias->num_axes());
        dim_ = bias_dim_ * inner_dim_;
        if(bottom[0] != top[0]){
            top[0]->ReshapeLike(*bottom[0]);
        }
        
        // bias_multiplier_的shape是[1，w],所有元素设置为1
        bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
        if(bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)){
            caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
        }
    }
    
    template <typename Dtype>
    void BiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const Dtype* bias_data = ((bottom.size()>1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
        
        if(bottom[0] != top[0]){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }
        
        for(int n=0; n<outer_dim_; ++n){
            caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_, 
                          inner_dim_, 1, Dtype(1), bias_data, 
                          bias_multiplier_.cpu_data(), Dtype(1), top_data);
            top_data += dim_;
        }
    }
    
    template <typename Dtype>
    void BiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        
        if(propagate_down[0] && bottom[0] != top[0]){
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
        }
        
        const bool bias_param = (bottom.size()==1);
        
        if((!bias_param && this->param_propagate_down_[0])){
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])->mutable_cpu_diff();
            
            bool accm = bias_param;
            for(int n = 0;  < outer_dim_; ++n){
                caffe_cpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
                              top_diff, bias_multiplier_.cpu_data(), Dtype(accum), bias_diff);
                top_dim += dim_;
                accum = true;
            }
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(BiasLayer);
#endif
    
    INSTANTIATE_CLASS(BiasLayer);
    REGISTER_LAYER_CLASS(Bias);
}
```

 1. 熟悉BiasParameter

    ```c++
    message BiasParameter {
      // The first axis of bottom[0] (the first input Blob) along which to apply
      // bottom[1] (the second input Blob).  May be negative to index from the end
      // (e.g., -1 for the last axis).
      //
      // For example, if bottom[0] is 4D with shape 100x3x40x60, the output
      // top[0] will have the same shape, and bottom[1] may have any of the
      // following shapes (for the given value of axis):
      //    (axis == 0 == -4) 100; 100x3; 100x3x40; 100x3x40x60
      //    (axis == 1 == -3)          3;     3x40;     3x40x60
      //    (axis == 2 == -2)                   40;       40x60
      //    (axis == 3 == -1)                                60
      // Furthermore, bottom[1] may have the empty shape (regardless of the value of
      // "axis") -- a scalar bias.
      optional int32 axis = 1 [default = 1];
    
      // (num_axes is ignored unless just one bottom is given and the bias is
      // a learned parameter of the layer.  Otherwise, num_axes is determined by the
      // number of axes by the second bottom.)
      // The number of axes of the input (bottom[0]) covered by the bias
      // parameter, or -1 to cover all axes of bottom[0] starting from `axis`.
      // Set num_axes := 0, to add a zero-axis Blob: a scalar.
      optional int32 num_axes = 2 [default = 1];
    
      // (filler is ignored unless just one bottom is given and the bias is
      // a learned parameter of the layer.)
      // The initialization for the learned bias parameter.
      // Default is the zero (0) initialization, resulting in the BiasLayer
      // initially performing the identity operation.
      optional FillerParameter filler = 3;
    }
    ```

 2. caffe_cpu_gemm在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
        Dtype* C);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template<>
    void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        const float alpha, const float* A, const float* B, const float beta,
        float* C) {
      int lda = (TransA == CblasNoTrans) ? K : M;
      int ldb = (TransB == CblasNoTrans) ? N : K;
      cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
          ldb, beta, C, N);
    }
    
    template<>
    void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        const double alpha, const double* A, const double* B, const double beta,
        double* C) {
      int lda = (TransA == CblasNoTrans) ? K : M;
      int ldb = (TransB == CblasNoTrans) ? N : K;
      cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
          ldb, beta, C, N);
    }
    ```

    cblas_sgemm和cblas_dgemm在OpenBlas定义如下，其功能为 C=alpha\*A\*B+beta\*C， 

    ```c++
    // 假设bottom[0]的形状是[n,c,h,w], axis=1, num_axes=2
    // outer_dim_ = n
    // bias_dim_ = c*h
    // inner_dim_ = w
    // dim_ = c*h*w
    
    // 那么A的形状为[c*h, 1]
    // B的形状为[1,w]
    // C的形状为[c*h,w]
    
    void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, OPENBLAS_CONST blasint ldc);
    
    void cblas_dgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, double *C, OPENBLAS_CONST blasint ldc);
    ```

 3. caffe_cpu_gemv在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
        Dtype* y);
    ```

    在math_functions.cpp中实例化如下：

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

    cblas_sgemv和cblas_dgemv在OpenBlas定义如下，其功能为 C=alpha\*A\*b+beta*C

    ```c++
    void cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
    
    void cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
    ```

    

**bias_layer.cu**

```c++
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    __global__ void BiasForward(const int n, const Dtype* in, const Dtype* bias,
                               const int bias_dim, const int inner_dim, Dtype* out){
        CUDA_KERNEL_LOOP(index, n){
            const int bias_index = (index/inner_dim) % bias_dim;
            out[index] = in[index] + bias[bias_index];
        }
    }
    
    template <typename Dtype>
    void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const int count = top[0]->count();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* bias_data = ((bottom.size() > 1) > bottom[1] : this->blobs_[0].get())->gpu_data();
        
        Dtype* top_data = top[0]->mutable_cpu_data();
        
        // 假设bottom[0]的形状是[n,c,h,w], axis=1, num_axes=2
        // outer_dim_ = n
        // bias_dim_ = c*h
        // inner_dim_ = w
        // dim_ = c*h*w
        BiasForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data,
                                                bias_data, bias_dim_, inner_dim_, top_data);
    }
    
    template <typename Dtype>
    void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0] && bottom[0]!=top[0]){
            const Dtype* top_diff = top[0]->gpou_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
        }
        
        const bool bias_param = (bottom.size()==1);
        if((!bias_param && propagate_down[1])) || (bias_param && this->param_propagate_down_[0]){
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bias_diff = (bias_param ? this->blobs_[0].get():bottom[1])->mutable_gpu_diff();
            bool accum = bias_param;
            for(int n=0; n<outer_dim_; ++n){
                caffe_gpu_gemv(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
                              top_diff, bias_multiplier_.gpu_data(), Dtype(accum), bias_diff);
                top_diff += dim_;
                accum = true;
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);
}
```

 1. caffe_gpu_gemv在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
        Dtype* y);
    ```

    caffe_gpu_gemv在math_functions.cu中实例化如下：

    ```c++
    template <>
    void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
        const int N, const float alpha, const float* A, const float* x,
        const float beta, float* y) {
      cublasOperation_t cuTransA =
          (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
      CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
          A, N, x, 1, &beta, y, 1));
    }
    
    template <>
    void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
        const int N, const double alpha, const double* A, const double* x,
        const double beta, double* y) {
      cublasOperation_t cuTransA =
          (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
      CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
          A, N, x, 1, &beta, y, 1));
    }
    ```

    cublasSegmv和cublasDgemv在cublas中定义如下，其功能为 Y=alpha\*A\*x+beta\*Y，A的形状是m*n (列优先)

    ```c++
    cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
        int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx,
        const float *beta, float *y, int incy)
    
    cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
        int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx,
        const double *beta, double *y, int incy)
    ```

    
