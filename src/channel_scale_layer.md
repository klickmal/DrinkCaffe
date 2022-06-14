# Channel Scale Layer
channel scale layer在channel_scale_layer.hpp，channel_scale_layer.cpp和channel_scale_layer.cu中声明实现。

**channel_scale_layer.hpp**

```c++
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

    template <typename Dtype>
    class ChannelScaleLayer: public Layre<Dtype>{
    public:
        explicit ChannelScaleLayer(const LayerParamter& param)
            : Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "ChannelScale";}
        virtual inline int ExactNumBottomBlobs() const {return 2;}
        
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
        
        Blob<Dtype> sum_multiplier_;
        bool do_forward_;
        bool do_backward_feature_;
        bool do_backward_scale_;
        bool global_scale_;
        Dtype max_global_scale_;
        Dtype min_global_scale_;
    };
	
}
```

**channel_scale_layer.cpp**

```c++
#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/channel_scale_layer.hpp"

namespace caffe{
    
    #define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))
    
    template <typename Dtype>
    void ChannelSclaeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top){
        CHECK(bottom[0]->num() == bottom[1]->num());
        CHECK(bottom[0]->count()/bottom[0]->channels() == bottom[1]->count());
        
        do_forward_ = this->layer_param_.channel_scale_param().do_forward();
        do_backward_feature_ = this->layer_param_.channel_scale_param().do_backward_feature();
        do_backward_scale_ = this->layer_param_.channel_scale_param().do_backward_scale();
        
        global_scale_ = this->layer_param_.channel_scale_param().global_scale();
        max_global_scale_ = this->layer_param_.channel_scale_param().max_global_scale();
        min_global_scale_ = this->layer_param_.channel_scale_param().min_global_scale();
        
        if(global_scale_){
            if(this->blobs_.size() > 0){
                LOG(INFO) << "Skipping parameter initialization";
            }
            else{
                this->blobs_.resize(1);
                this->blobs_[0].reset(new Blob<Dtype>({1}));
                this->blobs_[0]->mutable_cpu_data()[0] = -1;
            }

            if(this->layer_param_.param_size() == 0){
                ParamSpec* fixed_param_spec = this->layer_param_.add_param();
                fixed_param_spec->set_lr_mult(0.f);
                fixed_param_spec->set_decay_mult(0.f);
            }
            else{
                CHECK_EQ(this->layer_param_.param(0).lr_mult(), 0.f)
                    << "Cannot configure statistics as layer parameter.";
            }
        }
    }
    
    template <typename Dtype>
    void ChannelScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top){
        top[0]->ReshapeLike(*bottom[0]);
        if(global_scale_ && top.size() == 2){
            top[1]->Reshape({1});
        }
    }
    
    template <typename Dtype>
    void ChannelScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* scale_data = bottom[1]->cpu_data();
        
        Dtype* top_data = top[0]->mutable_cpu_data();
        
        if(do_forward_){
            if(global_scale_){
                int count = bottom[0]->count();
                Dtype* scale = this->blobs_[0]->mutable_cpu_data();
                scale[0] = std::min(scale[0], max_global_scale_);
                scale[1] = std::max(scale[0], min_global_scale_);
                
                if(top.size() == 2){
                    top[1]->mutable_cpu_data()[0] = scale[0];
                }
                caffe_cpu_scale(count, scale[0], bottom_data, top_data);
            }
        }
        else{
            int num = botttom[0]->num();
            int channels = bottom[0]->channels();
            int spatial_dim = bottom[0]->height() * bottom[0]->width();
            for(int n = 0; n < num; ++n){
                for(int s = 0; s< spatial_dim; ++s){
                    for(int c = 0; c < channels; ++c){
                        top_data[(n*channels + c) * spatial_dim + s] = scale_data[n * spatial_dim + s] *
                            bottom_data[(n * channels + c) * spatial_dim + s];
                    }
                }
            }
        }
        else
        {
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }
    }
    
    template <typename Dtype>
    void ChannelScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down, 
                                               const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* scale_data = bottom[1]->cpu_data();
        
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype* scale_diff = bottom[1]->mutable_cpu_diff();
        
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int spatial_dim = bottom[0]->height() * bottom[0]->width();
        
        if(propagate_down[0]){
            if(do_backward_feature_){
                for(int n = 0; n < num; ++n){
                    for(int s = 0; s < spatial_num; ++s){
                        for(int c = 0; c < channels; ++c){
                            bottom_diff[(n * channels + c) * spatial_dim + s] = 
                                scale_data[n * spatial_dim + s] * 
                                top_diff[(n * channels + c) * spatial_dim + s];
                        	}
                    	}
                	}
           	}
         	else{
               caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
        	}
    	}
    
    	caffe_set(num * spatial_dim, Dtype(0), scale_diff);
    
    	if(propagate_down[1] && do_backward_scale){
        	for(int n = 0; n < num; ++n){
            	for(int s = 0; s < spatial_dim; ++s){
                	for(int c = 0; c < channels; ++c){
                    scale_diff[n * spatial_dim + s] += bottom_data[(n * channels + c) * spatial_dim + s] *
                        top_diif[(n * channels + c) * spatial_dim + s];
                	}
            	}
        	}
    	}
	}

#ifdef CPU_ONLY
	STUB_GPU(ChannelScaleLayer);
#endif

	INSTANTIATE_CLASS(ChannelScaleLayer);
    REGISTER_LAYER_CLASS(ChannelScale);
   	
}

```

 1. caffe_cpu_scale在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

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

    

**channel_scale_layer.cu**

```c++
#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/channel_scale_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void kernel_channel_scale(const int num, const int channels, const int spatial_dm,
                                        Dtype alpha, const Dtype* data, const Dtype* norm_data, 
                                        Dtype beta, Dtype* output_data){
        CUDA_KERNEL_LOOP(index, num * channels * spatial_dim){
            int n = index / channels / spatial_dim;
            int s = index % spatial_dim;
            outptut_data[index] = alpha * data[index] * norm_data[n * saptial_dim + s] +
                beta * output_data[index];
        }
    }
    
    template <typename Dtype>
    __global__ void kernel_channel_sum(const int num, const int channels, const int spatial_dim,
                                      const Dtype* data, Dtype* sum_data){
        CUDA_KERNEL_LOOP(index, num * spatial_dim){
            int n  = index / spatial_dim;
            int s = index % spatial_dim;
            
            Dtype sum = 0;
            for(int c = 0; c < channels; ++c){
                sum += data[(n * channels + c) * spatial_dim + s];
            }
            sum_data[index] = sum;
        }
    }
    
    template <typename Dtype>
    void ChannelScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                              const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* scale_data = bottom[1]->gpu_data();
        
        Dtype* top_data = topp[0]->mutable_gpu_data();
        
        if(do_forward_){
            if(global_scale_){
                // ??????
                int count = bottom[0]->count();
                Dtype* scale = this->blobs_[0]->mutable_cpu_data();
                Dtype mean_norm = bottom[1]->asum_data() / (Dtype)bottom[1]->count();
                
                if(this->phase_ == TRAIN){
                    if(scale[0] < 0){
                        scale[0] = mean_norm;
                    }
                    else
                    {
                        scale[0] = scale[0]*0.99 + mean_norm*0.01;
                    }
                    scale[0] = std::min(scale[0], max_global_scale_);
                    scale[0] = std::max(scale[0], min_global_scale_);
                }
                
                if(top.size() == 2){
                    top[1]->mutable_cpu_data()[0] = scale[0];
                }
                caffe_gpu_scale(count, scale[0], bottom_data, top_data);
            }
            else{
                int num = bottom[0]->num();
                int channels = bottom[0]->channels();
                int spatial_dim = bottom[0]->height()*bottom[0]->width();
                kernel_channel_scale<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim),
                	CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, Dtype(1), bottom_data,
                                             scale_data, Dtype(0), top_data);
            }
        }
        else{
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }
    }
    
    template <typename Dtype>
    void ChannelScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                               const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff =top[0]->gpu_diff();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* scale_data = bottom[1]->gpu_data();
        
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        Dtype* scale_diff = bottom[1]->mutable_gpu_diff();
        
        int num = top[0]->num();
        int channels = top[0]->channels();
        int spatial_dim = bottom[0]->height() * bottom[0]->width();
        
        if(propagate_down[1]){
            if(do_backward_scale_){
                caffe_gpu_mul(bottom[0]->count(), top_diff, bottom_data, bottom_diff);
                kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(num*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(
                				num, channels, spatial_dim, bottom_diff, scale_diff); 
            }
            else{
                caffe_gpu_set(bottom[1]->count(), Dtype(0), scale_diff);
            }
        }
        
        if(propagate_down[0]){
            if(do_backward_feature_){
                kernel_channel_scale<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim), 
                	CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, Dtype(1), top_diff, 
                                             scale_data, Dtype(0), bottom_diff);
            }
            else{
                caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
            }
        }
    }
    
   	INSTANTIATE_LAYER_GPU_FUNCS(ChannelScaleLayer);
}
```

 1. caffe_gpu_scale在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_gpu_scale<float>(const int n, const float alpha, const float *x, float* y) {
    	CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
    	CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
    }
    
    template <>
    void caffe_gpu_scale<double>(const int n, const double alpha, const double *x, double* y) {
    	CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
    	CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
    }
    ```

 2. caffe_copy在math_functions.hpp中声明如下：

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
    ```

 3. caffe_gpu_set在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```c++
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
    	set_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, alpha, Y);
    }
    
    template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
    template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
    template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);
    ```

    
