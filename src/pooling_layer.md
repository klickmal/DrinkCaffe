# Pooling Layer
Pooling操作包括最大池化，平均池化。

pooling layer在pooling_layer.hpp, pooling_layer.cpp和pooling_layer.cu中实现

**pooling_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

   template <typename Dtype>
   class PoolingLayer: public Layer<Dtype>{
   public:
       explicit PoolingLayer(const LayerParameter& param)
           : Layer<Dtype>(param){}
       virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top);
       virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
       
       virtual inline const char* type() const {return "Pooling";}
       virtual inline int ExactNumBottomBlobs() const {return 1;}
       virtual inline int MinTopBlobs() const {return 1;}
       // MAX POOL layers can output an extra top blob for the mask;
  	   // others can only output the pooled inputs.
       virtual inline int MaxTopBlobs() const {
           return (this->layer_param_.pooling_param().pool() ==
                  PoolingParameter_PoolMethod_MAX) ? 2 : 1;
       }
       
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
       
       int kernel_h_, kernel_w_;
       int stride_h_, stride_w_;
       int pad_h_, pad_w_;
       int channels_;
       int height_, width_;
       int pooled_height_, pooled_width_;
       bool global_pooling_;
       Blob<Dtype> rand_idx_;
       Blob<int> max_idx_;
   };
}
```

 1. 定义pooling枚举类型

    ```c++
    enum PoolingParameter_PoolMethod {
      PoolingParameter_PoolMethod_MAX = 0,
      PoolingParameter_PoolMethod_AVE = 1,
      PoolingParameter_PoolMethod_STOCHASTIC = 2
    };
    ```

**pooling_layer.cpp**

```c++
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
   
using std::min;
using std::max;
    
    template <typename Dtype>
    void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*> top){
        PoolingParameter pool_param = this->layer_param_.pooling_param();
        if(pool_param.global_pooling()){
            CHECK(!(pool_param.has_kernel_size() ||
                  pool_param.has_kernel_h() || pool_param.has_kernel_w()))
                << "With Global_pooling: true Filter size cannot specified";
        }
        else
        {
            CHECK(!pool_param.has_kernel_size() !=
                 !(pooll_param.has_kernel_h() && pool_param.has_kernel_w()))
                << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
            CHECK(pool_param.has_kernel_size() ||
                (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
                << "For non-square filters both kernel_h and kernel_w are required.";
        }
        CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
              && pool_param.has_pad_w())
              || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
            << "pad is pad OR pad_h and pad_w are required.";
        CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
             && pool_param.has_stride_w())
              || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
            << "Stride is stride OR stride_h and stride_h and stride_w are required.";
        global_pooling_ = pool_param.global_pooling();
        if(global_pooling_){
            kernel_h_ = bottm[0]->height();
            kernel_w_ = bottom[0]->width();
        }
        else
        {
            if(pool_param.has_kernel_size()){
                kernel_h_ = kernel_w_ = pool_param.kernel_size();
            }
            else
            {
                kernel_h_ = pool_param.kernel_h();
                kernel_w_ = pool_param.kernel_w();
            }
        }
        
        CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
        CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
        if(!pool_param.has_pad_h()){
            pad_h_ = pad_w_ = pool_param.pad();
        }
        else{
            pad_h_ = pool_param.pad_h();
            pad_w_ = pool_param.pad_w();
        }
        if(!pool_param.has_stride_h()){
            stride_h_ = stride_w_ = pool_param.stride();
        }
        else{
            stride_h_ = pool_param.stride_h();
            stride_w_ = pool_param.stride_w();
        }
        
        if(global_pooling_){
            CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
                << "With Global_pooling: true; only pad = 0 and stride = 1";
        }
        if(pad_h_ != 0 || pad_w_ != 0){
            CHECK(this->layer_param_.pooling_param().pool()
                 == PoolingParameter_PoolMethod_AVE
                 || this->layer_param_.pooling_param().pool()
                 == PoolingParameter_PoolMethod_MAX)
                << "Padding implemented only for average and max pooling.";
            CHECK_LT(pad_h_, kernel_h_);
            CHECK_LT(pad_w_, kernel_w_);
        }
    }
    
    template <typename Dtype>
    void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
        CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
            << "corresponding to (num, channels, height, width)";
        channels_ = bottom[0]->channels();
        height_ = bottom[0]->height();
        width_ = bottom[0]->width();
        
        if(global_pooling_){
            kernel_h_ = bottom[0]->height();
            kernel_w_ = bottom[0]->width();
        }
        pooled_height_ = static_cast<int>(
            ceil(static_cast<float>(height_ + 2*pad_h_ - kernel_h_) / stride_h_)) + 1;
        pooled_width_ = static_cast<int>(
        	ceil(static_cast<float>(width_ + 2*pad_w_ - kernel_w_) / stride_w_)) + 1;
        if(pad_h_ || pad_w_){
            // If we have padding, ensure that the last pooling starts strictly
			// inside the image (instead of at the padding); otherwise clip the last.
            if((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_){
                --pooled_height_;
            }
            if((pooled_width_ -1) * strdie_w_ >= width_ + pad_w_){
                --pooled_width_;
            }
            CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
            CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
        }
        top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
        if(top.size() > 1){
            top[1]->ReshapeLike(*top[0]);
        }
        // If max pooling, we will initialize the vector index part.
        if(this->layer_param_.pooling_apram().pool() ==
           PoolingParameter_PoolMethod_MAX && top.size() == 1){
            max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
        }
        // If stochastic pooling, we will initialize the random index part.
        if(this->layer_param.pooling_param().pool() ==
           PoolingParameter_PoolingMethod_STOCHASTIC){
            rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
                             pooled_width_);
        }
    }
    
    template <typename Dtype>
    void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int top_count = top[0]->count();
        
         // We'll output the mask to top[1] if it's of size >1.
        const bool use_top_mask = top.size() > 1;
        int* mask = NULL;
        Dtype* top_mask = NULL;
        
        switch(this->layer_param_.pooling_param().pool()){
            case PoolingParameter_PoolMethod_MAX:
                if(use_top_mask){
                    top_mask = top[1]->mutable_cpu_data();
                    caffe_set(top_count, Dtype(-1), top_mask);
                }
                else
                {
                    mask = mask_idx_.mutable_cpu_data();
                    caffe_set(top_count, -1, mask);
                }
                caffe_set(top_count, Dtype(-FLT_MAX), top_data);
                
                for(int n = 0; n < bottom[0]->num(); ++n){
					for(int c = 0; c < channels_; ++c){
                        for(int ph = 0; ph < pooled_height_; ++ph){
                            for(int pw = 0; pw < pooled_width_; ++pw){
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, height_);
                                int wend = min(wstart + kernel_w_, width_);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);
                                const int pool_index = ph * pooled_width_ + pw;
                                for(int h = hstart; h < hend; ++h){
                                    for(int w = wstart; w < wend; ++w){
                                        const int index = h * width_ + w;
                                        if(bottom_data[index] > top_data[pool_index]){
                                            top_data[pool_index] = bottom_data[index];
                                            if(use_mask){
                                                top_mask[pool_index] = 
                                                    static_cast<Dtype>(index);
                                            }
                                            else{
                                                mask[pool_index] = index;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        // compute offset
                        bottom_data += bottom[0]->offset(0, 1);
                        top_data += top[0]->offset(0, 1);
                        if(use_top_mask){
                            top_mask += top[0]->offset(0, 1);
                        }
                        else{
                            mask += top[0]->offset(0,1);
                        }
                    }
                }
                break;
            case PoolingParameter_PoolMethod_AVE:
                for(int i = 0; i < top_count; ++i){
                    top_data[i] = 0;
                }
                for(int n =0; n < bottom[0]->num(); ++n){
                    for(int c = 0; c < channels_; ++c){
                        for(int ph = 0; ph < pooled_height_; ++ph){
                            for(int pw = 0; pw < pooled_width_; ++pw){
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw * stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, height_ + pad_h_);
                                int wend = min(wstart + kernel_w_, width_ + pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);
                                hend = min(hend, height_);
                                wend = min(wend, width_);
                                for(int h = hstart; h < hend; ++h){
                                    for(int w = wstart; w < wend; ++w){
                                        top_data[ph_ * pooled_width_ + pw_] += 
                                            bottom_data[h * width_ + w];
                                    }
                                }
                                top_data[ph * pooled_width_ + pw] /= pool_size;
                            }
                        }
                        // compute offset
                        bottom_data += bottom[0]->offset(0,1);
                        top_data += top[0]->offset(0,1);
                    }
                }
                break;
            case PoolingParameter_PoolMethod_STOCHASTIC:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Unknown pooling method.";
        }
    }
    
    template <typename Dtype>
    void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0]){
            return;
        }
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
        
        const bool use_top_mask = top.size() > 1;
        const int* mask = NULL;
        const Dtype* top_mask = NULL;
        switch(this->layer_param_.pooling_param().pool()){
            case PoolingParameter_PoolMethod_MAX:
                if(use_top_mask){
                    top_mask = top[1]->cpu_data();
                }
                else{
                    mask = max_idx_.cpu_data();
                }
                for(int n = 0; n < top[0]->num(); ++n){
                    for(int c = 0; c < channels_; ++c){
                        for(int ph = 0; ph < pooled_height; ++ph){
                            for(int pw = 0; pw < pooled_width; ++pw){
                                const int index = ph * pooled_width_ + pw;
                                const int bottom_index = use_top_mask ? 
                                    top_mask[index] : mask[index];
                                bottom_diff[bottom_index] += top_diff[index];
                            }
                        }
                        bottom_diff += bottom[0]->offset(0, 1);
                        top_diff += top[0]->offset(0, 1);
                        if(use_top_mask){
                            top_mask += top[0]->offset(0,1);
                        }
                        else
                        {
                            mask += top[0]->offset(0, 1);
                        }
                    }
                }
                break;
            case PoolingParameter_PoolMethod_AVE:
                for(int n = 0; n < top[0]->num(); ++n){
                    for(int c = 0; c < channels_; ++c){
                        for(int ph = 0; ph < pooled_height_; ++ph){
                            for(int pw = 0; pw < pooled_width_; ++pw){
                                int hstart = ph * stride_h_ - pad_h_;
                                int wstart = pw 8 stride_w_ - pad_w_;
                                int hend = min(hstart + kernel_h_, height_ + pad_h_);
                                int wend = min(wstart + kernel_w_, width_ + pad_w_);
                                int pool_size = (hend - hstart) * (wend - wstart);
                                hstart = max(hstart, 0);
                                wstart = max(wstart, 0);
                                hend = min(hend, height_);
                                wend = min(wend, width_);
                                for(int h = hstart; h < hend; ++h){
                                    for(int w = wstart; w < wend; ++w){
                                        bottom_diff[h * width_ + w] += 
                                           top_diff[ph * pooled_width_ + pw] / pool_size;
                                    }
                                }
                            }
                        }
                        bottom_diff += bottom[0]->offset(0, 1);
                        top_diff += top[0]->offset(0, 1);
                    }
                }
                break;
            case PoolingParameter_PoolMethod_STOCHASTIC:
                NOT_IMPLEMENTED;
                break;
            default:
                LOG(FATAL) << "Uknow pooling method."
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(PoolingLayer);
#endif
    
    INSTANTIATE_CLASS(PoolingLayer);
}

```

 1. caffe_set在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_set(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

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

**pooling_layer.cu**

```c++
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    __global__ void MaxPoolForward(const int nthreads, const Dtype* const bottom_data,
                                  const int num, const int channels, const int height,
                                  const int width, const int pooled_height,
                                  const int pooled_width, const int kernel_h,
                                  const int kernel_w, const int stride_h, 
                                  const int stride_w, const int pad_h, const int pad_w,
                                  Dtype* const top_data, int* mask, Dtype* top_mask){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int pw = index % pooled_width;
            const int ph = (index / pooled_width) % pooled_height;
            const int c = (index / pooled_width / pooled_height) % channels;
            const int n = index / pooled_width / pooled_height / channels;
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            
            const int hend = min(hstart + kernel_h, height);
            const int wend = min(wstart + kernel_w, width);
            
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            Dtype maxval = -FLT_MAX;
            int maxidx=  -1;
            const Dtype* const bottom_slice = bottom_data + (n*channels+c)*height*width;
            for(int h = hstart; h < hend; ++h){
                for(int w = wstart; w < wend; ++w){
                    if(bottom_slice[h * width + w] > maxval){
                        maxidx = h * width + w;
                        maxval = bottom_slice[maxidx];
                    }
                }
            }
            top_data[index] = maxval;
            if(mask){
                mask[index] = maxidx;
            }
            else
            {
                top_mask[index] = maxidx;
            }
        }
    }
    
    template <typename Dtype>
    __global__ void AvePoolForward(const int nthreads, const Dtype* const bottom_data,
                                  const int num, const int channels, const int height,
                                  const int width, const int pooled_height,
                                  const int pooled_weight, const int kernel_h,
                                  const int kernel_w, const int stride_h, 
                                  const int stride_w, const int pad_h, 
                                  const int pad_w, Dtype* const top_data){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int pw = index % pooled_width;
            const int ph = (index / pooled_width) % pooled_height;
            const int c = (index / pooled_width / pooled_height) % channels;
            const int n = index / pooled_width / pooled_height/ chanels;
            
            int hstart = ph * stride_h - pad_h;
            int wstart = pw * stride_w - pad_w;
            int hend = min(hstart + kernel_h, height + pad_h);
            int wend = min(wstart + kernel_w, width + pad_w);
            
            const int pool_size = (hend-hstart)*(wend-wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            
            hend = min(hend, height);
            wend = min(wend, width);
            
            Dtype aveval = 0;
            const Dtype* const bottom_slice = 
                bottom_data + (n*channels + c)*height*width;
            for(int h = hstart; h < hend; ++h){
                for(int w = wstart; w < wend; ++w){
                    aveval += bottom_slice[h * width + w];
                }
            }
            top_data[index] = aveval / pool_size;
        }
    }
    
    template <typename Dtype>
    __global__ void GlobalAvePoolForward(const int spatial_dim,
                                        const Dtype* bottom_data,
                                        Dtype* top_data){
        __shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];
        unsigned int tid = threadIdx.x;
        buffer[tid] = 0;
        __syncthreads();
        
        for(int j = tid; j < spatial_dim; j+= blockDim.x){
            buffer[tid] += bottom_data[blockIdx.x*spatial_dim + j];
        }
        __syncthreads();
        
        for(int i = blockDim.x / 2; i > 0; i>>1){
            if(tid < i){
                buffer[threadIdx.x] += buffer[threadsIdx.x + i];
            }
            __syncthreads();
        }
        
        if(tid == 0){
            top_data[blockIdx.x] = buffer[0] / spatial_dim;
        }
    }
    
    template <typename Dtype>
    void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        int count = top[0]->count();
        
        const bool use_top_mask = top.size() > 1;
        int* mask = NULL;
        Dtype* top_mask = NULL;
        
        switch(this->layer_param_.pooling_param().pool()){
            case PoolingParameter_PoolMethod_MAX:
                if(use_top_mask){
                    top_mask = top[1]->mutable_gpu_data();
                }
                else{
                    mask = max_idx_.mutable_gpu_data();
                }
             	MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), 
                	CAFFE_CUDA_NUM_THREADS>>>(
                    count, bottom_data, bottom[0]->num(), channels_, height_,
                    width_, pooled_height_, pooled_width_, kernel_h_,
                    kernel_w_, stride_h_, stride_w_, pad_h, pad_w_, top_data,
                    mask, top_mask);
        		break;
            case PoolingParameter_PoolMethod_AVE:
                if(this->layer_param_.pooling_param().global_pooling()){
                    GlobalAvePoolingForward<Dtype><<<bottom[0]->count(0,2), 
                    		CAFFE_CUDA_NUM_THREADS>>>(bottom[0]->count(2), 
                                                      bottom_data, top_data);
                }
                else
                {
                    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), 
                    		CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, 
                                    bottom[0]->num(), channels_, height_,
                                    width_, pooled_height_, pooled_width_, kernel_h_,
                                    kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
                                    top_data);
                }
                break;
            default:
                LOG(FATAL) << "Unknown pooling method."
        }
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    __global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
                                   const int* const mask, const Dtype* const top_mask,
                                   const int num, const int channels, const int height,
                                   const int width, const int pooled_height,
                                   const int pooled_width, const int kernel_h,
                                   const int kernel_w, const int stride_h,
                                   const int stride_w, const int pad_h,
                                   const int pad_w, Dtype* const bottom_diff){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int w = index % width;
            const int h = (index / width) % height;
            const int c = (index / width / height) % channels;
            const int n = index / width / height/ channels;
            const int phstart = 
                (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
            const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
            const int pwstart = 
                (h + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
            const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
            
            Dtype gradient = 0;
            const int offset = (n * channels + c) * pooled_height * pooled_width;
            const Dtype* const top_diff_slice = top_diff + offdet;
            
            if(mask){
                const int* const mask_slice = mask + offset;
                for(int ph = phstart; ph < phend; ++ph){
                    for(int pw = pwstart; pw < pwend; ++pw){
                        if(mask_slice[ph * pooled_width + pw] == h * width + w){
                            gradient += top_diff_slice[ph * pooled_width + pw];
                        }
                    }
                }
            }
            else
            {
                const Dtype* const top_mask_slice = top_mask + offset;
                for(int ph = phstart; ph < phend; ++ph){
                    for(int pw = pwstart; pw < pwend; ++pw){
                        if(top_mask_slice[ph * pooled_width + pw] == h*width + w){
                            gradient += top_diff_slice[ph * pooled_width + pw];
                        }
                    }
                }
            }
            bottom_diff[index] = gradient;          
        }
    }
    
    template <typename Dtype>
    __global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
                                   const int num, const int channels, const int height,
                                   const int width, const int pooled_height, 
                                   const int pooled_width, const int kernel_h, 
                                   const int kernel_w, const int stride_h,
                                   const int stride_w, const int pad_h, const int pad_w,
                                   Dtype* const bottom_diff){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int w = index % width + pad_w;
            const int h = (index / width) % height + pad_h;
            const int c = (index / width / height) % channels;
            const int n = index / width / height / channels;
            
            const int phstart = (h < kernel_h) ? 0 : (h - kernel_h)/stride_h + 1;
            const int phend = min(h / stride_h + 1, pooled_height);
            const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w)/stride_w + 1;
            const int pwend = min(w / stride_w + 1, pooled_width);
            
            Dtype gradient = 0;
            const Dtype* const top_diff_slice =
                top_diff + (n * channels + c) * pooled_height * pooled_width;
            
            for(int ph = phstart; ph < phend; ++ph){
                for(int pw = pwstart; pw < pwend; ++pw){
                    int hstart = ph * stride_h - pad_h;
                    int wstart = pw * stride_w - pad_w;
                    int hend = min(hstart + kernel_h, height + pad_h);
                    int wend = min(wstart + kernel_w, width + pad_w);
                    int pool_size = (hend - hstart) * (wend - wstart);
                    
                    gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
                }
            }
            bottom_diff[index] = gradient;
        }
    }
    
    template <typename Dtype>
    __global__ void GlobalAvePoolBackward(const int nthreads, const int spatial_dim,
                                         const Dtype* top_diff, Dtype* bottom_diff){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int n = index / spatial_dim;
            bottom_diff[index] = top_diff[n] / spatial_dim;
        }
    }
    
    template <typename Dtype>
    void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0]){
            return;
        }
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        
        caffe_gpu_set(count, Dtype(0), bottom_diff);
        
        const bool use_top_mask = top.size() > 1;
        const int* mask = NULL;
        const Dtype* top_mask = NULL;
        switch(this->layer_param_.pooling_param().pool()){
            case PoolingParameter_PoolMethod_MAX:
                if(use_top_mask){
                    top_mask = top[1]->gpu_data();
                }
                else
                {
                    mask = max_idx_.gpu_data();
                }
             	MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), 
                		CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, mask, top_mask,
                                                 top[0]->num(), channels_,
                                                 height_, width_, pooled_height_,
                                                 pooled_width_, kernel_h_, kernel_w_,
                                                 stride_h_, stride_w_, pad_h_, pad_w_,
                                                 bottom_diff);
                break;
            case PoolingParameter_PoolMethod_AVE:
                if(this->layer_param_.pooling_param().global_pooling()){
                    GlobalAvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), 
                    	CAFFE_CUDA_NUM_THREADS>>>(count, bottom[0]->count(2), top_diff,
                                                 bottom_diff);
                }
                else
                {
                    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), 
                    	CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, top[0]->num(),
                                                 channels_, height_, width_, 		
                                                  pooled_height_, pooled_width_, 
                                                  kernel_h, kernel_w_, stride_h, 	 
                                                  stride_w_, pad_h_, pad_w_,bottom_diff);
                }
                break;
            default:
                LOG(FATAL) << "Unknown pooling method.";
        }
        CUDA_POST_KERNEL_CHECK;
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);
}
```

