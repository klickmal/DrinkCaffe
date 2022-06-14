# Normalize Layer
normalize layer在normalize_layer.hpp, normalize.cpp和normalize.cu中实现

**normalize_layer.hpp**

```c++
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class NormalizeLayer : public Layer<Dtype>{
    public:
        explicit NormalizeLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vecotr<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Normalize";}
        virtual inline int ExactNumBottomBlobs() const {return 1;}
        virtual inline int MinTopBlobs() const {return 1;}
        virtual inline int MaxTopBlobs() const {return 2;}
    
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
        // norm_中储存所有通道数的均方根
        Blob<Dtype> sum_multiplier_, squared_, norm;
        std::string normalize_type_;
        bool fix_gradient_;
        bool bp_norm_; // 这个bp_norm_是啥？
    };
}
```

**normalize_layer.cpp**

```c++
#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe{
    
#define sign(x) ((Dtype(0) < (x)) - ((x) - Dtype(0)))
    
    template <typename Dtype>
    void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top){
        normalize_type_ = this->layer_param_.normalize_param().normalize_type();
        fixed_gradient_ = this->layer_param_.normalize_param().fix_gradient();
        bp_norm_ = this->layer_param_.normalize_param().bp_norm() && top.size() == 2;
    }
    
    template <typename Dtype>
    void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                       bottom[0]->height(), bottom[0]->width());
        squared_.Reshape(bottom[0]->num(), bottom[0]->channels(),
                        bottom[0]->height(), bottom[0]->width());
        if(top.size() == 2){
            top[1]->Reshape(bottom[0]->num(), 1, 
                            bottom[0]->height(), bottom[0]->width());
        }
        norm_.Reshape(bottom[0]->num(), 1,
                     bottom[0]->height(), bottom[0]->width());
    }
    
    template <typename Dtype>
    void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        Dtype* square_data = squared_.mutable_cpu_data();
        
        Dtype* norm_data = (top.szie() == 2) ? top[1]->mutable_cpu_data() : 
        					norm_.mutable_cpu_data();
        
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int spatial_dim = bottom[0]->height() * bottom[0]->width();
        
        if(normalize_type_ == "L2"){
            caffe_sqrt<Dtype>(num * channels * spatial_dim, bottom_data, square_data);
            for(int n = 0; n < num; n++){
                for(int s = 0; s < spatial_dim; s++){
                    norm_data[n * spatial_dim + s] = Dtype(0);
                    // 在channel上进行normalize
                    for(int c = 0; c < channels; ++c){
                        norm_data[n * spatial_dim + s] += 
                            square_data[(n * channels + c) * spatial_dim + s];
                    }
                    norm_data[n * spatial_dim + s] += 1e-6;
                    norm_data[n * spatial_dim + s] = 
                        sqrt(norm_data[n * spatial_dim + s]);
                    for(int c = 0; c < channels; c++){
                        top_data[(n * channels + c) * spatial_dim + s] =
                            bottom_data[(n * channels + c) * spatial_dim + s] /
                            norm_data[n * spatial_sim + s];
                    }
                }
            }
        }
        else if(normalize_type_ == 'L1'){
            caffe_abs<Dtype>(num * channels * spatial_dim, bottom_data, square_data);
            
            for(int n = 0; n < num; ++n){
                for(int s = 0; s < spatial_dim; ++s){
                    norm_data[n * spatial_dim + s] = Dtype(0);
                    for(int c = 0; c < channels; ++c){
                        norm_data[n * spatial_dim + s] += 
                            square_data[(n * channels + c) * spatial_dim + s];
                    }
                    norm_data[n * spatial_dim + s] += 1e-6;
                    norm_data[n * spatial_dim + s] = norm_data[n * spatial_dim + s];
                    
                    for(int c = 0; c < channels; ++c){
                        top_data[(n * channels + c) * spatial_dim + s] = 
                            bottom_data[(n * channels + c) * spatial_dim + s] / 
                            	norm_data[n * spatial_dim + s];
                    }
                }
            }
        }
        else{
            NOT_IMPLEMENTED;
        }
    }
    
    template <typename Dtype>
    void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* square_data = squared_.cpu_data();
        const Dtype* norm_data = (top.size() == 2) ? 
            				top[1]->cpu_data() : norm.cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int spatial_dim = bottom[0]->height() * bottom[0]->width();
        
        if(propagate_down[0]){
            if(normalize_type_ == "L2"){
                for(int n = 0; n < sum; ++n){
                    for(int s = 0; s < spatial_dim; ++s){
                        Dtype a = caffe_gpu_strided_dot(channeles, top_data + 
                                  n * channels * spatila_dim + s, spatial_dim,
                                  top_diff + n * channels * spatial_dim + s,
                                  spatial_dim);
                        for(int c = 0; c < channels; c++){
                            bottom_diff[(n * channels + c) * spatial_dim + s] = 
                                (top_diff[(n * channels + c) * spatial_dim + s] -
                                 top_data[(n * channels + c) * spatial_dim + s] * a) / 
                                norm_data[n * spatial_dim + s];
                        }
                    }
                }
            }
            else if(normalize_type_ == "L1"){
                for(int n = 0; n < num; ++n){
                    for(int s = 0; s < spatial_dim; ++s){
                        Dtype a = caffe_cpu_strided_dot(channels, top_data + 
                                n * channels * spatial_dim + s, spatial_dim,
                                top_diff + n * channels * spatial_dim + s, spatial_dim);
                    	for(int c = 0; c < channels; ++c){
                            bottom_diff[(n * channels + c) * spatial_dim + s] = 
                                (top_diff[(n * channels + c) * spatail_dim + s] - 
                                sign(bottom_data[(n * channels + c) * spatial_dim + s]) * a) / 
                                norm_data[n * spatial_dim + s]; 
                        }
                    }
                }
            }
            else{
                NOT_IMPLEMENTED;
            }
        }
        if(bp_norm_){
            const Dtype* norm_diff = top[1]->cpu_diff();
            if(normalize_type_ == "L2"){
                for(int n = 0; n < num; ++n){
                    for(int s = 0; s < spatial_dim; ++s){
                        for(int c = 0; c < channels; ++c){
                            bottom_diff[(n*channels + c) * spatial_dim + s] +=
                                norm_diff[n * spatial_dim + s] * 
                                top_data[(n * channels + c) * spatial_dim + s];
                        }
                    }
                }
            }
            else if(normalize_type_=='L1'){
                for(int n = 0; n < num; ++n){
                    for(int s = 0; s < spatial_num; ++s){
                        for(int c = 0; c < channels; ++c){
                            bottom_diff[(n * channels + c) * spatial_dim + s] +=
                                norm_diff[n * spatial_dim + s] * 
                                sign(bottom_data[(n * channels + c) * spatial_dim + s]);
                        }
                    }
                }
            }
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(NormalizeLayer);
#endif
    
    INSTANTIATE_CLASS(NormalizeLayer);
    REGISTER_LAYER_CLASS(Normalize);
}

```

 1. caffe_sqr在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_sqr(const int N, const Dtype* a, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_sqr<float>(const int n, const float* a, float* y) {
    	vsSqr(n, a, y);
    }
    
    template <>
    void caffe_sqr<double>(const int n, const double* a, double* y) {
    	vdSqr(n, a, y);
    }
    ```

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

 2. caffe_abs在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_abs(const int n, const Dtype* a, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

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

 3. caffe_cpu_strided_dot在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
        const Dtype* y, const int incy);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    float caffe_cpu_strided_dot<float>(const int n, const float* x,
                                       const int incx, const float* y, const int incy) {
    	return cblas_sdot(n, x, incx, y, incy);
    }
    
    template <>
    double caffe_cpu_strided_dot<double>(const int n, const double* x,
                                         const int incx, const double* y, const int incy) {
    	return cblas_ddot(n, x, incx, y, incy);
    }
    ```

    ```c++
    float  cblas_sdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
    
    double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);
    ```

    

**normalize_layer.cu**

```c++
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void kernel_channel_sum(const int num, const int channels, 
                        const int spatial_dim, Dtype epsilon, const Dtype* data,
                        Dtype* norm_data){
        CUDA_KERNEL_LOOP(index, num * spatial_dim){
            int n = index / spatial_dim;
            int s = index % saptial_dim;
            Dtype sum = 0;
            for(int c = 0; c < channels; ++c){
                sum += data[(n * channels + c) * spatial_dim + s];
            }
            norm_data[index] = sum + epsilon;
        }
    }
    
    template <typename Dtype>
    __global__ void kernel_channel_scale(const int num, const int channels,
                         const int spatial_dim, Dtype alpha, const Dtype* data,
                         const Dtype* norm_data, Dtype beta, Dtype* output_data){
        CUDA_KERNEL_LOOP(index, num * channels * spatial_dim){
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            output_data[index] = alpha * data[index] * norm_data[n * spatial_dim + s] +
                beta * output_data[index];
        }
    }
    
    template <typename Dtype>
    __global__ void kernel_channel_self_scale(const int num, const int channels, 
                            const int spatial_dim, const Dtype* norm_data,
                            Dtype* input_output_data){
        CUDA_KERNEL_LOOP(index, num * channels * spatial_dim){
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            
            intput_output_data[index] *= norm_data[n * spatial_dim + s];
        }
    }
    
    template <typename Dtype>
    __global__ void kernel_channel_div(cosnt int num, const int channels, 
                                       const int spatial_dim, Dtype alpha, 
                                      const Dtype* data, const Dtype* norm_data,
                                      Dtype beta, Dtype* output_data){
        CUDA_KERNEL_LOOP(index, num * channels * spatial_dim){
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            output_data[index] = alpha * data[index] / norm_data[n * spatial_dim + s]
                + beta * output_data[index];
        }
    }
    
    template <typename Dtype>
    __global__ void kernel_self_div(const int num, const int channels, 
                                   const int spatial_dim, const Dtype* norm_data,
                                   Dtype* input_output_data){
        CUDA_KERNEL_LOOP(index, num*channels*spatial_dim){
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            input_output_data[index] /= norm_data[n * spatial_dim + s];
        }
    }
    
    template <typename Dtype>
    __global__ void kernel_channel_dot(const int num, const int channels,
                                      const int spatial_dim, const Dtype* data_1,
                                      const Dtype* data_2, Dtype* channel_dot){
        CUDA_KERNEL_LOOP(index, num*spatial_dim){
            int n = index / spatial_dim;
            int s = index % spatial_dim;
            Dtype dot = 0;
            for(int c = 0; c < channels; ++c){
                dot += (data1[(n*channels + c)*spatial_dim +s] *
                       data2[(n*channels + c)*spatial_dim + s]);
            }
            channel_dot[index] = dot;
        }
    }
    
    template <typename Dtype>
   	__global__ void kernel_sign(const int count, const Dtype* input, Dtype* sign_out){
        CUDA_KERNEL_LOOP(index, count){
            sign_out[index] = (Dtype(0) < input[index] - (input[index] < Dtype(0)));
        }
    }
    
    template <typename Dtype>
    void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        Dtype* square_data = squared_.mutable_gpu_data();
        Dtype* norm_data = (top.size() == 2) ? top[1]->mutable_gpu_data() : 
        					norm_.mutable_gpu_data();
        int num = bottom[0]->num();
        int channels = bottom[0]->channel();
        int spatial_dim = bottom[0]->height() * bottom[0]->width();
        if(normalize_type == 'L2'){
            caffe_gpu_powx(num * channels * spatial_dim, bottom_data, 
                           Dtype(2), square_data);
            kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num*spatial_dim),
            			CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim,
                        1e-12, square_data, norm_data);
            caffe_gpu_powx(num * spatial_dim, norm_data, Dtype(0.5), norm_data);
            kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
            			CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, 
                                                 Dtype(1), bottom_data, norm_data, 
                                                  Dtype(0), top_data);
        }
        else if(normalize_type == "L1"){
            caffe_gpu_abs(num*channels*spatial_dim, bottom_data, square_data);
            kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num*saptial_dim),
            			CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim,
                                                 1e-6, square_data, norm_data);
            kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
            			CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, Dtype(1),
                                             bottom_data, norm_data, Dtype(0), top_data);
        }
        else
        {
            NOT_IMPLEMENTED;
        }
    }
    
    template <typename Dtype>
    void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* top_data = top[0]->gpu_data();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* square_data = squared_.mutable_gpu_data();
        const Dtype* norm_data = (top.size() == 2) ? top[1]->gpu_data() :
        							norm_.gpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        Dtype* temp_diff = norm_.mutable_gpu_diff();
        
        int num = top[0]->num();
        int channels = top[0]->channels();
        int spatial_dim = bottom[0]->height() * bottom[0]->width();
        
        if(propagate_down[0]){
            kernel_channel_dot<Dtype><<<CAFFE_GET_BLOCKS(num*spatial_dim),
            					CAFFE_CUDA_NUM_THREADS>>>(num, channelsm, spatial_dim,
                                                         top_data, top_diff, temp_diff);
            if(normalize_type_ == "L2"){
                kernel_channel_scale<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                				CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim,
                                                         Dtype(1), top_data, temp_diff,
                                                         Dtype(0), bottom_diff);
            }
            else if(normalize_type_ == "L1"){
                kernel_sign<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                				CAFFE_CUDA_NUM_THREADS>>>(num*channels*spatial_dim,
                                                         bottom_data, square_data);
                kernel_channel_scale<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                				CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim,
                                           				Dtype(1), square_data, temp_diff,
                                                        Dtype(0), bottom_diff);
            }
            else{
                NOT_IMPLEMENTED;
            }
            
            caffe_gpu_sub(num * channels * spatial_dim, top_diff, 
                          bottom_diff, bottom_diff);
            if(fix_gradient_){
                
            }
            else{
                kernel_channel_self_div<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, norm_data, bottom_diff);
            }
        }
        if(bp_norm_){
            const Dtype* norm_diff = top[1]->gpu_diff();
            if(normalize_type_ == "L2"){
                kernel_channel_scale<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                		CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, Dtype(1),
                                            top_data, norm_diff, Dtype(1), bottom_diff);
            }
            else if(normalize_type_ == "L1"){
                if(!propagate_down[0]){
                    kernel_sign<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                    	CAFFE_CUDA_NUM_THREADS>>>(num*channels*spatial_dim, bottom_data, 													square_data);
                }
                kernel_channel_scale<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                		CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, Dtype(1),
                                          square_data, norm_diff, Dtyep(1), bottom_diff);
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);
}
```

 1. caffe_gpu_powx在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void powx_kernel(const int n, const Dtype* a, const Dtype alpha, Dtype* y) {
    	CUDA_KERNEL_LOOP(index, n) {
    		y[index] = pow(a[index], alpha);
    	}
    }
    
    template <>
    void caffe_gpu_powx<float>(const int N, const float* a, const float alpha, float* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	powx_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, alpha, y);
    }
    
    template <>
    void caffe_gpu_powx<double>(const int N, const double* a, const double alpha, double* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	powx_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, alpha, y);
    }
    ```

 2. caffe_gpu_sub在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void sub_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
    	CUDA_KERNEL_LOOP(index, n) {
    		y[index] = a[index] - b[index];
    	}
    }
    
    template <>
    void caffe_gpu_sub<float>(const int N, const float* a, const float* b, float* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	sub_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
    }
    
    template <>
    void caffe_gpu_sub<double>(const int N, const double* a, const double* b, double* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	sub_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
    }
    ```

    
