# PRelu Layer
PRelu即带参数的Relu, 如果参数取一个很小的固定值，则PRelu退化为Leaky Relu. PRelu计算公式如下:
$$
y=\left\{\begin{matrix}
x, \ if \ x\ge 0  \\
ax, \ otherwise
\end{matrix}\right.
$$
在此，a是可学习参数。

prelu layer在prelu_layer.hpp, prelu_layer.cpp和prelu_layer.cu中实现

**prelu_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    class PReLULayer: public NeuronLayer<Dtype>{
    public:
        explicit PReluLayer(const LayerParameter& param)
            :NeuronLayer<Dtype>(param){}
        
        virtual void LayerSetup(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "PRelu";}
        
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
        
        bool channel_shared_; // Whether or not slope parameters are shared across channels
        Blob<Dtype> multiplier_; // dot multiplier for backward computation of params
        Blob<Dtype> backward_buff_; // temporary buffer for backward computations
        Blob<Dtype> bottom_memory_; // memory for in-place computation
        
    };
}
```

**prelu_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

namespace caffe{
    
	template <typename Dtype>
    void PReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        CHECK_GE(bottom[0]->num_axes(), 2)
            << "Number of axes of bottom blob must be >=2.";
        PReLUParameter prelu_param = this->layer_param().prelu_param();
        
        int channels = bottom[0]->channels();
        channel_shared_ = prelu_param.channel_shared();
        
        if(this->blobs_.size() > 0){
            LOG(INFO) << "Skipping parameter initialization";
        } 
        else {
            this->blobs_.resize(1);
            if(channel_shared_){
                this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
            }
            else {
                this->blobs_[0].reset(new Bob<Dtype>(vector<int>(1, channels)));
            }
            shared_ptr<Filler<Dtype>> filler;
            if(prelu_param.has_filler()){
                filler.reset(GetFiller<Dtype>(prelu_param.filler()));
            }
            else{
                FillerParametr filler_param;
                filler_param.set_type("constant");
                filler_param.set_value(0.25);
                filler.reset(GetFiller<Dtype>(filler_param));
            }
            filler->Fill(this->blobs_[0].get());
        }
        if(channel_shared_){
            CHECK_EQ(this->blobs_[0]->cpunt(), 1)
                << "Negative slop size is inconsistent with prototxt config";
        }
        else{
            CHECK_EQ(this->blobs_[0]->count(), channels)
                << "Negative slop size is inconsistent with prototxt config";
        }
        
        this->param_propagate_down_.resize(this->blobs_.size(), true);
        multiplier_.Reshape(vector<int>(1, bottom[0]->count(1)));
        backward_buff_.Reshape(vector<int>(1, bottom[0]->count(1)));
        caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
    }
    
    template <typename Dtype>
    void PReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top){
        CHECK_GE(bottom[0]->num_axes(), 2)
            << "Number of axes of bottom blob must be >=2.";
        top[0]->ReshapeLike(*bottom[0]);
        
        if(bottom[0] == top[0]){
            bottom_memory_.ReshapeLike(*bottom[0]);
        }
    }
    
    template <typename Dtype>
    void PReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        const int dim = bottom[0]->count(2);
        const int channels = bottom[0]->channels();
        
        const Dtype* slope_data = this->blobs_[0]->cpu_data();
        
        if(bottom[0] == top[0]){
            caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
        }
        
        const int div_factor = channel_shared_ ? channels : 1;
        for(int i=0; i < count; ++i){
            int c = (i / dim) % channels / div_factor;
            top_data[i] = std::max(bottom_data[i], Dtype[0] 
                          + slope_data[c]*std::min(bottom_data[i], Dtype(0));
        }
    }
    
    template <typename Dtype>
    void PReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* slope_data = this->blobs_[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        
        const int count = bottom[0]->count();
        const int dim = bottom[0]->count(2);
        const int channels = bottom[0]->channels();
        
        if(top[0] == bottom[0]){
            bottom_data = bottom_memory_.cpu_data();
        }
        
        const int div_fractor = channel_shared_ ? channels : 1;
        
        if(this->param_propagate_down_[0]){
            Dtype* slope_diff = this->blobs_[0]->mutable_cpu_diff();
            for(int i=0; i<count; ++i){
                int c = (i/dim) % channels / div_factor;
                slop_diff[c] += top_diff[i]*bottom_data[i]*(bottom_data[i] <= 0);
            }
        }
        
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            for(int i=0; i<count; ++i){
                int c = (i/dim) % channels / div_factor;
                bottom_diff[i] = top_diff[i]*((bottom_diff[i]>0) 
                                              + slope_data[c]*(bottom[i] <= 0));
            }
        }
    }
                                   
#ifdef CPU_ONLY
   	STUB_GPU(PReLULayer);
#endif
    
    INSTANTIATE_CLASS(PReLULayer);
    REGISTER_LAYER_CLASS(PReLU);
}

```

 1. PRelu_Parameter

    ```c++
    message PReLUParameter {
      // Parametric ReLU described in K. He et al, Delving Deep into Rectifiers:
      // Surpassing Human-Level Performance on ImageNet Classification, 2015.
    
      // Initial value of a_i. Default is a_i=0.25 for all i.
      optional FillerParameter filler = 1;
      // Whether or not slope parameters are shared across channels.
      optional bool channel_shared = 2 [default = false];
    }
    ```

    

**prelu_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void PReLUForward(const int n, const int channels, const int dim,
          const Dtype* in, Dtype* out, const Dtype* slope_data, const int div_factor){
        CUDA_KERNEL_LOOP(index, n){
            int c = (index / dim) % channels / div_factor;
           	out[index] = in[index] > 0 ? in[index] : in[index]*slope_data[c];
        }
    }
    
    template <typename Dtype>
    __global__ void PReLUBackward(const int n, const int channels, const int dim,
             const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff, 
             const Dtype* slope_data, const int div_factor){
        CUDA_KERNEL_LOOP(index, n){
            int c = (index / dim) % channels / div_factor;
            out_diff[index] = in_diff[index] * ((in_data[index]>0) 
                                                + (in_data[index]<=0)*slope_data[c]);
        }
    }
    
    template <typename Dtype>
    __global__ void PReLUParamBackward(const int n, const int rows, const int rowPitch,
                   const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff){
        CUDA_KERNEL_LOOP(index, n){
            out_diff[index] = in_diff[index] * in_data[index] * (in_data[index]<=0);
            
            // rows = n
            // rowPitch = c*h*w
            for(int k=1; k<rows; ++k){
                out_diff[index] += in_diff[index + k*rowPitch]
                    *in_data[index + k*rowPitch]*(in_data[index + k*rowPitch] <= 0);
            }
        }
    }
    
    template <typename Dtype>
    void PReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();
        const int dim = bottom[0]->count(2);
        
        const int channels = bottom[0]->channels();
        const Dtype* slope_data = this->blobs_[0]->gpu_data();
        const int div_factor = channel_shared_ ? channels : 1;
        
        if(top[0] == bottom[0]){
            caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
        }
        
        PReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
       			count, channels, dim, bottom_data, top_data, slope_data, div_factor);
    	CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    void PReLULayer<Dtype>::Backward_gpu(const vecor<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        const int count = bottom[0]->count();
        const int dim = bottom[0]->count(2);
        const int channels = bottom[0]->channels();
        
        if(top[0] == bottom[0]){
           	bottom_data = bottom_memory_.gpu_data();
        }
        
        // propagate to param
        if(this->param_propagate_down_[0]){
            Dtype* slope_diff = this->blobs_[0]->mutable_gpu_diff();
            int cdim = channels * dim;
            
            PReLUParamBackward<Dtype><<<CAFFE_GET_BLOCKS(cdim), CAFFE_CUDA_NUM_THREADS>>>(
                			   cdim, bottom[0]->num(), top[0]->offset(1),
                               top_diff, bottom_data, backward_buff_.mutable_gpu_diff());
            CUDA_POST_KERNEL_CHECK;
            
            if(channel_shared_){
                Dtype sum;
                caffe_gpu_dot<Dtype>(channels * dim, backward_buff_.gpu_diff(),
                                    multiplier_.gpu_data(), &dsum);
                caffe_gpu_add_scalar(this->blobs_[0]->count(), Dtype(dsum), slope_diff);
            }
            else{
                caffe_gpu_gemv<Dtype>(CblasNoTrans, channels, dim, 1, 	
                                   backward_buff_.gpu_diff(), multiplier_.gpu_data(), 1, slope_diff);
            }
        }
        
        // propagate to bottom
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const Dtype* slope_data = this->blobs_[0]->gpu_data();
            
            int div_factor = channel_shared_ ? channels : 1;
            PReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            			count, channels, dim, top_diff, bottom_data, 
                		bottom_diff, slope_data, div_factor);
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(PReLULayer);
}
```

 1. caffe_gpu_dot在math_functions.hpp中声明如下：

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

 2. caffe_gpu_add_scalar在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cu中进行实例化

    ```c++
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

 3. caffe_gpu_gemv在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
        Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```C++
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

    
