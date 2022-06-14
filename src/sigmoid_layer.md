# Sigmoid Layer
sigmoid layer在sigmoid_layer.hpp，sigmoid_layer.cpp和sigmoid_layer.cu中声明实现。sigmoid激活函数如下：
$$
y = \frac{1 }{1+e^{-x} }
$$
sigmoid激活函数可以用tanh函数表示，caffe就是用该关系式实现sigmoid激活函数：
$$
tanh(x) = \frac{1-e^{-2x} }{1+e^{-2x}} = \frac{2}{1+e^{-2x} } -1 = 2sigmoid(x)-1
$$
除法函数求导公式：
$$
\begin{pmatrix}
\frac{u}{v}
\end{pmatrix}^{'} =\frac{u^{'} v-v^{'}u }{v^{2} }
$$
因此sigmoid激活函数求导：
$$
y^{'}=\frac{0-(-e^{-x} )}{(1+e^{-x} )^{2} }  = \frac{e^{-x} +1 -1 }{(1+e^{-x} )^{2}} = \frac{1}{1+e^{-x}}-\frac{1}{(1+e^{-x} )^{2}}=y(1-y) 
$$
**sigmoid_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{

    template <typename Dtype>
    class SigmoidLayer: public NeuronLayer<Dtype>{
    public:
        explicit SigmoidLayer(const LayerParameter& param)
            :NeuronLayer<Dtype>(param){}
        virtual inline const char* type() const {return "Sigmoid";}
    
    protected:
    	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
        virtual vois Backeard_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
    };
}
```

**sigmoid_layer.cpp**

```c++
#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe{
    
   	template<typename Dtype>
   	inline Dtype sigmoid(Dtype x){
        return 0.5*tanh(0.5*x) + 0.5;
    }
    
    template <typename Dtype>
    void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->cpu_data();
        
        const int count = bottom[0]->count();
        for(int i=0; i<count; ++i){
            top_data[i] = sigmoid(bottom_data[i]);
        }
    }
    
    template <typename Dtype>
    void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_data = top[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            
            for(int i=0; i<count; ++i){
                const Dtype sigmoid_x = top_data[i];
                bottom_diff[i] = top_diff[i]*sigmoid_x*(1-sigmoid_x);
            }
        }
    }
    
#ifdef CPU_ONLY
	STUB_GPU(SigmoidLayer);
#endif
    
    INSTANTIATE_CLASS(SigmoidLayer);
}

```

 1. 思考为什么用tanh实现sigmoid


**sigmoid_layer.cu**

```c++
#include <cmath>
#include <vector>

#include "caffe/layers/sigmoid_layer.hpp"

namespace caffe{
    
   	template <typename Dtype>
  	__global__ void SigmoidForward(const int n, const Dtype* in, Dtype* out){
       CUDA_KERNEL_LOOP(index, n){
           out[index] = 0.5 * tanh(0.5*in[index]) + 0.5;
       }
   	}
    
    template <typename Dtype>
    void SigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();
        
        SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        					count, bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    __global__ void SigmoidBackward(const int n, const Dtype* in_diff,
                                   const Dtype* out_data, Dtype* out_diff){
        CUDA_KERNEL_LOOP(index, n){
            const Dtype sigmoid_x = out_data[index];
            out_diff[index] = in_diff[index]*sigmoid_x*(1-sigmoid_x);
        }
    }
    
    template <typename Dtype>
    void SigmoidLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          cosnt vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_data = top[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            
            SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THRERADS>>>(
            						count, top_diff, top_data, bottom_diff);
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    INSTANTIATE_LAYER_GPUY_FUNCS(Sigmoidlayer);
}
```

