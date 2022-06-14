# Relu Layer
relu layer在relu_layer.hpp, relu_layer.cpp和relu_layer.cu中实现。Relu是LeakyRelu的特殊情况，LeakyRelu激活函数如下：
$$
y = \left\{\begin{matrix}
x, \ if \ x\ge 0\\
negative\_slope*x, \ otherwise
\end{matrix}\right. 
$$
**relu_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    class ReLULayer: public NeuronLayer<Dtype>{
    public:
        explicit ReLULayer(cont LayerParameter& param):
        	NeuronLayer<Dtype>(param){}
        
        virtual inline const char* type() const {return "RELU";}
        
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

**relu_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe{
    
	template <typename Dtype>
    void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        
        //negative_slope是Leak Relu的参数，默认为0，就是普通的Relu函数。
		//Leaky ReLU是为解决“ReLU死亡”问题的尝试。
		//一般的ReLU中当x<0时，函数值为0。禠eaky ReLU则是给出一个很小的负数梯度值，比如0.01。
		//Leaky Relu公式如下 f(x) = max(x, 0) + alpha*min(x, 0) 其中alpha就是下面的代码中参数		 		
        //negative_slope
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
        
        for(int i=0; i<count; ++i){
            top_data[i] = std::max(bottom_data[i], Dtype(0)) 
                + negative_slope*std::min(bottom_data[i], Dtype(0));
        }
    }
    
    template <typename Dtype>
    void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagte_down[0]){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            
            Dtype negative_slop = this->layer_param_.relu_param().negative_slope();
            for(int i=0; i<count; ++i){
                bottom_diff[i] = top_diff[i]*((bottom_data[i]>0) 
                                 + negative_slope*(bottom_data[i]<=0));
            }
        }
    }
    
#ifdef CPU_ONLY
	STUB_GPU(ReLULayer);
#endif
    
    INSTANTIATE_CLASS(ReLULayer);
}

```

**relu_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/relu_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void ReLUForward(const int n, const Dtype* in, Dtype* out, 
                                Dtype negative_slop){
        CUDA_KERNEL_LOOP(index, n){
            out[index] = in[index]>0 ? in[index] : in[index]*negative_slop;
        }
    }
    
    template <typename Dtype>
    void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        const int count = bottom[0]->count();
        Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
        
        ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        					count, bottom_data, top_data, negative_slop);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    __global__ void ReLUBackward(const int n, const Dtype* in_diff, const Dtype* in_data, 
                                Dtype* out_diff, Dtype negative_slope){
        CUDA_KERNEL_LOOP(index, n){
            out_diff[index] = in_diff[index]*((in_data[index]>0) 
                                              + (in_data[index]<=0)*negative_slope);
        }
    }
    
    template <typename Dtype>
    void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* bottom_data = bottom[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            
            Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
            
            ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            				count, top_diff, bottom_data, bottom_diff, negative_slope);
            
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);
}
```

