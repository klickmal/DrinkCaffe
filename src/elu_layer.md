# Elu Layer

elu layer在源文件**elu_layer.hpp、elu_layer.cpp、elu_layer.cu**中实现。
$$
f(x) = \begin{cases} 
x,  & \text{if }\text{ x > 0} \\
\alpha*(exp(x)-1), & \text{if }\text{ x <= 0}
\end{cases}
$$
求导：
$$
f'(x) = \begin{cases} 
1,  & \text{if }\text{ x > 0} \\
\alpha*exp(x) = f(x) + \alpha, & \text{if }\text{ x <= 0}
\end{cases}
$$
**elu_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
	template <typename Dtype>
    class ELULayer: public NeuronLayer<Dtype>{
    public: 
      	explicit ELULayer(const LayerParameter& param):
        	NeuronLayer<Dtype>(param){}
        
      	virtual inline const char* type() const {return "ELU";}
     
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



**elu_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe{    
    
    template <typename Dtype>
    void ELULayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->cpu_data();
       
       Dtype* top_data = top[0]->mutable_cpu_data();
       const int count = bottom[0]->count();
       Dtype* alpha = this->layer_param_.elu_param().alpha();
        
       for(int i = 0; i < count; ++i){
          top_data[i] = std::max(bottom_data[i], Dtype(0)) + 
              alpha*(exp(std::min(bottom_data[i], Dtype(0))) - Dtype(1));
       }
    }
    
    template <typename Dtype>
    void ELULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){ 
        if(propagate_down[0]){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_data = top[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            
            Dtype alpha = this->layer_param_.elu_param().alpha();
            
            for(int i = 0; i < count; ++i){
				bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0) + 
                                                (alpha + top_data[i]) *(bottom_data[i] <= 0)); 
            }
        }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(ELULayer);
    #endif
    
    INSTANTIATE_CLASS(ELULayer);
    REGISTER_LAYER_CLASS(ELU);
}
```

**elu_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/elu_layer.hpp"

namespace caffe{
   
    template <typename Dtype>
    __global__ void ELUForward(const int n, const Dtype* in, Dtype* out, Dtype alpha){
        CUDA_KERNEL_LOOP(index, n){
            out[index] = in[index] > 0 ? in[index] : alpha * (exp(in[index]) - 1);
        }
    }
    
    template <typename Dtype>
    void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        const int count = bottom[0]->count();
        Dtype alpha = this->layer_param_.elu_param().alpha();
        
        ELUFoward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, 	
                                                                             top_data, alpha);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    __global__ void ELUBackward(const int n, const Dtype* in_diff, const Dtype* out_data, 
                                const Dtype* in_data, Dtype* out_diff, Dtype alpha)	{
        CUDA_KERNEL_LOOP(index, n){
            out_diff[index] = in_data[index] > 0 ? in_diff[index] : 
            				in_diff[index] * (out_data[index] + alpha);
        }
    }
    
    template <typename Dtype>
    void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vecrtor<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* bottom_data = bottom[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            const Dtype* top_data = top[0]->gpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            Dtype alpha = this->layer_param_.elu_param().alpha();
            
            ELUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                                  count, top_diff, top_data, bottom_data, bottom_diff, alpha);
            CUDA_POST_KERNEL_CHECK;
        }
    }
     INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);
}
```

