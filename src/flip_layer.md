# Flip Layer

flip层将blob沿着width和height进行反转。flip layer在源文件**flip_layer.hpp、flip_layer.cpp、flip_layer.cu**中实现。

**flip_layer.hpp**

```c++
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
	template <typename Dtype>
    class FlipLayer: public Layer<Dtype>{
    public: 
      	explicit FlipLayer(const LayerParameter& param):
        	Layer<Dtype>(param){}
      	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
                              const vector<Blob<Dtype>*>& top);
      	virtual inline const char* type() const {return "Flip";}
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
        
        bool flip_width_;
        bool flip_heighht_;
    };
}
```

FlipLayer是Layer的派生类，头文件声明了必要的成员函数。

**flip_layer.cpp**

```c++
#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/flip_layer.hpp"
#include "caffe/net.hpp"

namespace caffe{
	template <typename Dtype>
    void FlipLayer<Dtype>::LayerSetUp(const vectror<Blob<Dtype>*>& bottom,
                                       const vector<Blob<type>*>& top){
        flip_width_ = this->layer_param_.flip_param().flip_width();
        flip_height_ = this->layer_param_.flip_param().flip_height();
    }
    
    template <typename Dtype>
    void FlipLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top){
        top[0]->ReshapeLike(*bottom[0]);
    }
    
    template <typename Dtype>
    void FlipLayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->cpu_data();
       Dtype* top_data = top[0]->mutable_cpu_data();
       
       int num = bottom[0]->num();
       int channels = bottom[0]->channels();
       int width = bottom[0]->width();
       int height = bottom[0]->height();
        
       for(int n = 0; n < num; n++){
           for(int c = 0; c < channels; c++){
               for(int h = 0; h < height; h++){
                   for(int w = 0; w < width; w++){
                       // 在H*W上左右上下互换 
                       top_data[(((n*channels + c)*height + h)*width) + w] = 
                           bottom_data[(((n*channels + c)*height + (flip_height ? (height - 1 -h) ：h)) * 							width) + (flip_width_ ? (width - 1 - w):w)];
                   }
               }
           }
       }
    }
    
    template <typename Dtype>
    void FlipLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom){
      const Dtype* top_diff = top[0]->cpu_diff();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      
      int num = bottom[0]->num();
      int channels = bootm[0]->channels();
      
      int width = bottom[0]->with();
      int height = bottom[0]->height();
        
      if(propagate_down[0]){
          for(int n = 0; n < num; n++){
              for(int c = 0; c < channels(); c++){
                  for(int h = 0; h < height; h++){
				  	for(int w = 0; w < width; w++){
                        // 梯度也是在H*W上进行对调
                        bottom_diff[(((n*channels + c)*height + h)*width) + w] = 
                            top_diff[(((n*channels + c) * height + (flip_height_ ? (height - 1 - h) : h)) * 							width) + (flip_width_ ? (width - 1 - w) : w)];
                    }
                  }
              }
          }
      }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(FlipLayer)
    #endif
    
    INSTANTIATE_CLASS(FlipLayer);
    REGISTER_LAYER_CLASS(Flip); 
}
```


**flip_layer.cu**

```c++
#include <vector>

#include "caffe/layers/flip_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void FlipKernel(const int num, const int channels, const int height, const int width,
                              const Dtype* src_data, Dtype* target_data, bool flip_height, bool flip_width){
        CUDA_KERNEL_LOOP(index, num*channels*height*width){
            int n = index / (channels * height * width);
            int cs = index % (channels * height * width);
            int c = cs / (height * width);
            int s = cs % (height * width);
            int h = s / width;
            int w = s % width;
            
            target_data[(((n * channels + c)*height + h)*width)+w] = 
                src_data[(((n*channels + c)*height + (flip_height ? (height - 1 - h))*width) 
                          + (flip_width ? (width-1-w):w)];
        }
    }
    
    template <typename Dtype>
    void FlipLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->gpu_data();
       Dtype* top_data = top[0]->mutable_gpu_data();
       
       int num = bottom[0]->num();
       int channels = bottom[0]->channels();
       int width = bottom[0]->width();
       int height = bottom[0]->height();
       
       FlipKernel<Dtype><<<CAFFE_GET_BLOCKS(num*channels*height*width), CAFFE_CUDA_NUM_THREADS>>>(
       num, channels, height, width, bottom_data, top_data, flip_height_, flip_width_);

    }
    
    template <typename Dtype>
    void FlipLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        
        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        int width = bottom[0]->width();
        int height = bottom[0]->height();
        
        if(propagate_down[0]){
            FlipKernel<Dtype><<<CAFFE_GET_BLOCKS(num*channels*height*width), CAFFE_CUDA_NUM_THREADS>>>(
            num, channels, height, width, top_diff, bottom_diff, flip_height_, flip_width_);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(FlipLayer);
}
```
