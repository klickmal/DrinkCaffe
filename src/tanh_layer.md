# Tanh Layer
tanh layer计算公式如下：
$$
y = \frac{e^{x}-e^{-x}  }{e^{x}+e^{-x} }
$$

tanh layer在源文件tanh_layer.hpp, tanh_layer.cpp和tanh_layer.cu中实现

**tanh_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
template <typename Dtype>
class TanHLayer: public NeuronLayer<Dtype>{
    public:
        explicit TanHLayer(const LayerParameter& param):
            NeuronLayer<Dtype>(param){}
    	
        virtual inline const char* type() const {return "TanH";} 

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
    	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<Blob<Dtype>*>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
    };
}
```

**tanh_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
     	const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        
        for(int i=0; i<count; ++i){
            top_data[i] = tanh(bottom_data[i]);
        }
    }
    
    template <typename Dtype>
    void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_data = top[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            
            const int count = bottom[0]->count();
            Dtype tanhx;
            
            for(int i=0; i<count; ++i){
                tanhx = top_data[i];
                bottom_diff[i] = top_diff[i]*(1-tanhx*tanhx);
            }
        }
    }

#ifdef CPU_ONLY
	STUP_GPU(TanHLayer);
#endif
    
    INSTANTIATE_CLASS(TanHLayer);
}

```



**tanh_layer.cu**

```c++
#include <vector>

#include "caffe/layers/tanh_layer.hpp"

namespace caffe{

   template <typename Dtype>
   __global__ void TanHForward(const int n, const Dtype* in, Dtype* out){
       CUDA_KERNEL_LOOP(index, n){
           out[index] = tanh(in[index]);
       }
   }
   
   template <typename Dtype>
   void TanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->gpu_data();
       Dtype* top_data = top[0]->mutable_gpu_data();
       const int count = bottom[0]->count();
       
       TanHForward<Dtype><<<CAFEE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
       						count, bottom_data, top_data);
       
       CUDA_POST_KERNEL_CHECK;
   }
    
   template <typename Dtype>
   __global__ void TanHBackward(const int n, const Dtype* in_diff, 
                               const Dtype* out_data, Dtype* out_diff){
       CUDA_KERNEL_LOOP(index, n){
           Dtype tanhx = out_data[index];
           out_diff[index] = in_diff[index] *(1-tanhx*tanhx);
       }
   }
    
   template <typename Dtype>
   void TanHLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Bob<Dtype>*>& down){
       if(propagate_down[0]){
           const Dtype* top_data = top[0]->gpu_data();
           const Dtype* top_diff = top[0]->gpu_diff();
           Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
           const int count = bottom[0]->count();
           TanHBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
           					count, top_diff, top_data, bottom_diff);
        	CUDA_POST_KERNEL_CHECK;
       }
   }
    
    INSTANTIATE_LAYER_GPU_FUNCS(TanHLayer);
}
```

