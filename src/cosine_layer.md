# Cosine Layer

cosine layer在源文件**cosine_layer.hpp、cosine_layer.cpp、cosine_layer.cu**中实现。

**cosine_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
	template <typename Dtype>
    class CosineLayer: public NeuronLayer<Dtype>{
    public: 
      	explicit CosineLayer(const LayerParameter& param):
        	NeuronLayer<Dtype>(param){}
        
      	virtual inline const char* type() const {return "Cosine";}
        virtual inline int MinBottomBlobs const {return 1;}
        virtual inline int ExactNumTopBlobs const {return 1;}
     
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
    };
}
```

**cosine_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/cosine_layer.hpp"

namespace caffe{    
    
    template <typename Dtype>
    void CosineLayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->cpu_data();
       
       Dtype* top_data = top[0]->mutable_cpu_data();
       const int count = bottom[0]->count();
       
       for(int i = 0; i < count; ++i){
          top_data[i] = cos(bottom_data[i]);
       }
    }
    
    template <typename Dtype>
    void CosineLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
   		
     	const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        
        const int count = bottom[0]->count();
        
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            
            for(int i = 0; i < count; ++i){
                bottom_diff[i] = -sin(bottom_data[i]);
            }
        }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(CosineLayer);
    #endif
    
    INSTANTIATE_CLASS(CosineLayer);
    REGISTER_LAYER_CLASS(Cosine);
}
```

**cosine_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/cosine_layer.hpp"

namespace caffe{
   
    template <typename Dtype>
    __global__ void ConsineForward(const int n, const Dtype* in, Dtype* out){
        CUDA_KERNEL_LOOP(index, n){
            out[index] = cos(in[index]);
        }
    }
    
    template <typename Dtype>
    __global__ void CosineBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff)	{
        CUDA_KERNEL_LOOP(index, n){
            out_diff[index] = in_diff[index] * -1 * sin(in_data[index]);
        }
    }
    
    template <typename Dtype>
    void CosineLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        const int count = bottom[0]->count();
        
        CosineFoward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, 	
                                                                                 top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    void CosineLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vecrtor<Blob<Dtype>*>& bottom){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        
        const int count = bottom[0]->count();
        
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            
            CosineBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                                  count, top_diff, bottom_data, bottom_diff);
            CUDA_POST_KERNEL_CHECK;
        }
    }
     INSTANTIATE_LAYER_GPU_FUNCS(CosineLayer);
}
```

