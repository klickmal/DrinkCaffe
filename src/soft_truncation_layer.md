# Soft Truncation Layer
soft truncation layer在源文件soft_truncation_layer.hpp, soft_truncation_layer.cpp和soft_truncation_layer.cu中实现。

soft truncation层计算公式为：
$$
y=1-e^{\frac{-x}{c} }
$$
soft truncation层求导：
$$
y^{'} =\frac{e^{\frac{-x}{c} } }{c} =\frac{1-y}{c} 
$$
**soft_truncation_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{

    template <typename Dtype>
    class SoftTruncationLayer: public NeuronLayer<Dtype>{
    public:
        explicit SoftTruncationLayer(const LayerParameter& param):
                 NeuronLayer<Dtype>(param){}

        virtual inline const char* type() const{return "SoftTruncation";}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype>*>& top);
    };
    
}
```

**soft_truncation_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/soft_truncation_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    void SoftTruncationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        
       	Dtype c = this->layer_param_.soft_truncation_param().c();
        
        for(int i = 0; i < count; ++i){
           top_data[i] = 1 - exp(bottom_data[i]/(-c))
        }
    }
    
    // 牛皮，复用top_data
	// f(x) = 1-exp(-x/c)
	// f'(x) = exp(-x/c)/c = (1-f(x))/c
    template <typename Dtype>
    void TruncationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
           const Dtype* top_data = top[0]->cpu_data();
           const Dtype* top_diff = top[0]->cpu_diff();
           Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
           const int count = bottom[0]->count();
           
           Dtype c = this->layer_param_.soft_truncation_param().c();
           
           for(int i=0; i < count; ++i){
               bottom_diff[i] = top_diff[i]*(1-top_data[i])/c;
           }
        }
    }

#ifdef CPU_ONLY
	STUP_GPU(SoftTruncationLayer);
#endif
    
    INSTANTIATE_CLASS(SoftTruncationLayer);
    REGISTER_LAYER_CLASS(SoftTrauncation);
}

```

**soft_truncation_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/soft_truncation_layer.hpp"

namespace caffe{
    
   template <typename Dtype>
   __global__ void SoftTruncationForward(const int n, const Dtype* in, Dtype* out, Dtype c){
       CUDA_KERNEL_LOOP(index, n){
           out[index] = 1- exp(in[index]/(-c));
       }
   }
    
   template <typename Dtype>
   void SoftTruncationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                                               const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->gpu_data();
       Dtype* top_data = top[0]->mutable_gpu_data();
       const int count = bottom[0]->count();
       
       Dtype c = this->layer_param_.soft_truncation_param().c();
       
       SoftTruncationForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
                                                 bottom_data, top_data, c);
       
       CUDA_POST_KERNEL_CHECK;
   }
   
    template <typename Dtype>
    __global__ void SoftTruncationBackward(const int n, const Dtype* in_diff, const Dtype* in_data,
                                          Dtype* out_diff, Dtype c){
        CUDA_KERNEL_LOOP(index, n){
            out_diff[index] = in_diff[index]*(1-in_data[index])/c;
        }
    }
    
    template <typename Dtype>
    void SoftTruncationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down,
                                                 const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_data = top[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            Dtype c = this->layer_param_.soft_truncation_param().c();
            
            SoftTruncationBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            							count, top_diff, top_data, bottom_diff, c);
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(SoftTruncationLayer);
}
```

