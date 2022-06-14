# Smooth L1 Loss Layer
Smooth L1 Loss Layer在源文件smooth_L1_layer.hpp, smooth_L1_layer.cpp和smooth_L1_layer.cu中实现。Smooth L1 Loss的计算公式如下所示：
$$
y_{smooth}  = \left\{\begin{matrix}
0.5(x\sigma)^{2}, \ if \ \begin{vmatrix}
x
\end{vmatrix} < 1 \\
\begin{vmatrix}
x
\end{vmatrix} -\frac{0.5}{\sigma^{2} } , \ otherwise
\end{matrix}\right.
$$
smooth_L1_Loss是FasterRCNN提出来的计算距离的loss，文章中提到对噪点更加鲁棒。输入四个Bottom，分别是predict, target, inside_weight 和 outside_weight。与论文并不完全一致，代码中实现的是更加general的版本，公式为：
$$
y_{smooth}^{new} =\omega _{out}*y_{smooth}(\omega _{in} x), x_{i} = p_{i}-gt_{i}
$$
**smooth_L1_loss_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe{
	template <typename Dtype>
	class SmoothL1LossLayer::public LossLayer<Dtype>{
    public:
        explicit SmoothL1LossLayer(const LayerParameter& param):
        					LossLayer<Dtype>(param), diff_(){}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const { return "SmoothL1Loss";}
        
        virtual inline int ExactNumBottomBlobs() const {return -1;}
        virtual inline int MinBottomBlobs() const {return 2;}
        virtual inline int MaxBottomBlobs() const {return 4;}
        
    	virtual inline bool AllowForceBackward(const int bottom_index) const {return true;}
     
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
        
        Blob<Dtype> diff_;
        Blob<Dtype> errors_;
        Blob<Dtype> ones_;
        bool has_weights_;
        Dtype sigma2_;
    };
}
```

**smooth_L1_loss_layer.cpp**

```c++
#include "caffe/layers/smooth_L1_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top){
        SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
        sigma2_ = loss_param.sigma()*loss_param.sigma();
        
        has_weights_ = (bottom.size()>=3);
        if(has_weights_){
            CHECK_EQ(bottom.size(), 4) << "If weights are used, must specify both "
                "inside and outside weights";
        }
    }
    
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top){
        LossLayer<Dtype>::Reshape(bottom, top);
        CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
        CHECK_EQ(bottom[0]->height(), bottom[1]->height());
        CHECK_EQ(bottom[0]->width(), bottom[1]->width());
        
        if(has_weights_){
            CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
            CHECK_EQ(bottom[0]->height(), bottom[2]->height());
            CHECK_EQ(bottom[0]->width(), bottom[2]->width());
            CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
            CHECK_EQ(bottom[0]->height(), bottom[3]->height());
            CHECK_EQ(bottom[0]->width(), bottom[3]->width());
        }
        diff_.Reshape(bottom[0]->num(), bottom[0]->channles(), 
                     bottom[0]->height(), bottom[0]->width());
        errors_.Reshape(bottom[0]->num(), bottom[0]->channles(), 
                       bottom[0]->height(), bottom[0]_.width());
        ones_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                     bottom[0]->height(), bottom[0]->width());
        
        for(int i=0; i<bottom[0]->count(); ++i){
            ones_.mutable_cpu_data()[i] = Dtype(1);
        }
    }
    
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
        NOT_IMPLEMENTED;
    }
    
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom){
        NOT_IMPLEMENTED;
    }
    
    #ifdef CPU_ONLY
   	STUB_GPU(SmoothL1LossLayer);
    #endif
    
    INSTANTIATE_CLASS(SmoothL1LossLayer);
    REGISTER_LAYER_CLASS(SmoothL1Loss);
}
```

**smooth_L1_loss_layer.cu**

```c++
#include "caffe/layers/smooth_L1_loss.layer"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void SmoothL1Forward(const int n, const Dtype* in, Dtype* out, Dtype sigma2){
        CUDA_KERNEL_LOOP(index, n){
            Dtype val = in[index];
            Dtype abs_val = abs(val);
            
            if(abs_val < 1.0 / sigma2){
                out[index] = 0.5*val*val*sigma2;
            }
            else{
                out[index] = abs_val - 0.5/sigma2;
            }
        }
    }
    
    template <typename Dtype>
    void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
        int count = bottom[0]->count();
        caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                     diff_.mutable_gpu_data());
        if(has_weights_){
            caffe_gpu_mul(count, bottom[2]->gpu_data(), diff_.gpu_data(), diff_.mutable_gpu_data());
        }
        
        SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
                                             diff_.gpu_data(), errors_.mutable_gpu_data(), sigma2);
        CUDA_POST_KERNEL_CHECK;
        
        if(has_weights_){
            caffe_gpu_mul(count, bottom[3]->gpu_data(), errors_.gpu_data(), errors_.mutable_gpu_data());
        }
        
        Dtype loss;
        caffe_gpu_dot(count, ones_.gpu_data(), errors_.gpu_data(), &loss);
        top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
    }
    
    template <typename Dtype>
    __global__ void SmoothL1Backward(const int n, const Dtype* in, Dtype* out, Dtype sigma2){
        CUDA_KERNEL_LOOP(index, n){
            // f'(x) = sigma * sigma * x         if |x| < 1 / sigma / sigma
  			//       = sign(x)                   otherwise
            Dtype val = in[index];
            Dtype abs_val = abs(val);
            
            if(abs_val < 1.0/sigma2){
                out[index] = sigma2*val;
            }
            else{
                out[index] = (Dtype(0)<val)-(val-Dtype(0));
            }
        }
    }
    
    template <typaname Dtype>
    void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom){
        int count = diff_.count();
        SmoothL1LossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        		count, diff_.gpu_data(), diff_.mutable_gpu_data(), sigma2);
        CUDA_POST_KERNEL_CHECK;
        
        for(int i=0; i<2; ++i){
            if(propagate_down[i]){
                const Dtype sign = (i==0) ? 1 : -1;
                const Dtype alpha = sign*top[0]->cpu_diff()[0]/bottom[i]->num();
                
                caffe_gpu_axpby(count, alpha, diff_.gpu_data(), Dtype(0), bottom[i]->mutable_gpu_diff());
                
                if(has_weights_){
                     // Scale by "inside" weight
                    caffe_gpu_mul(count, bottom[2]->gpu_data(), 
                                  bottom[i]->gpu_diff(), bottom[i]->mutable_gpu_diff());
                    // Scale by "outside" weight
                    caffe_gpu_mul(count, bottom[3]->gpu_data(),
                                 bottom[i]->gpu_diff(), bottom[i]->mutable_gpu_diff());
                }
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);
}
```

 1. caffe_gpu_sub在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void sub_kernel(const int n, const Dtype* a,
        const Dtype* b, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = a[index] - b[index];
      }
    }
    
    template <>
    void caffe_gpu_sub<float>(const int N, const float* a, const float* b, float* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
    }
    
    template <>
    void caffe_gpu_sub<double>(const int N, const double* a, const double* b, double* y) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
    }
    ```

 2. caffe_gpu_mul在math_functions.hpp声明如下：

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

 3. caffe_gpu_dot在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <>
    void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
        float* out) {
      CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
    }
    
    template <>
    void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
        double * out) {
      CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
    }
    ```

 4. caffe_gpu_axpby在math_functions.hpp声明如下：

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
    
    void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
    	if (X != Y) {
    		CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));
        }
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
