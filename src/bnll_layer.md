# BNLL Layer

BNLL layer在源文件**bnll_layer.hpp、bnll_layer.cpp、bnll_layer.cu**中实现。BNLL layer计算公式：
$$
y = \left\{\begin{matrix}
x + log(1+e^{-x} ), \  if \ x > 0 \\
log(1+e^{x}), \  otherwise
\end{matrix}\right.
$$
**bnll_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namesapce caffe{
	template <typename Dtype>
	class BNLLLayer : public NeuronLayer<Dtype> {
	public:
		explicit BNLLLayer(const LayerParameter& param)
			: NeuronLayer<Dtype>(param) {}

		virtual inline const char* type() const { return "BNLL"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
        
       virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    };
}
```



**bnll_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe{    
    
    const float kBNLL_THRESHOLD = 50.;
    
    template <typename Dtype>
    void BNLLLayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->cpu_data();
       Dtype* top_data = top[0]->mutable_cpu_data();
       const int count = bottom[0]->count();
        
       for(int i = 0; i < count; ++i){
           top_data[i] = bottom[i] > 0 ? bottom_data[i] + log(1. + exp(-bottom_data[i])) 
               : log(1. + exp(bottom_data[i]));
       }
    }
    
    template <typename Dtype>
    void BNLLLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<type>*>& bottom){
        if(propagate_down[0]){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            const int count = bottom[0]->count();
            
            Dtype expval;
            for(int i = 0; i < count; ++i){
                expval = exp(std::min(bottom_data[i], Dtype(kBNLL_THRESHOLD)));
                bottom_diff[i] = top_diff[i] * expval / （expval + 1.）
            }
        }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(BNLLLayer)
    #endif
    
    INSTANTIATE_CLASS(BNLLLayer);
    REGISTER_LAYER_CLASS(BNLL); 
}
```
1. 对BNLL公式进行求导，并统一导数形式：

   $$
   y^{'} =\frac{e^{x} }{1+e^{x} } 
   $$
   

**bnll_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/bnll_layer.hpp"

namespace caffe{
    __constant__ float kBNLL_THRESHOLD = 50.;
    
    template <typename Dtype>
    __global__ void BNLLForward(const int n, const Dtype* in, Dtype* out){
        CUDA_KERNEL_LOOP(index, n){
            out[index] = in[index] > 0 ? in[index] + log(1. + exp(-in[index])) :
            log(1. + exp(in[index]));
        }
    }
    
    template <typename Dtype>
    void BNLLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                                      const vector<Blob<Dtype>*>& top>){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        
        BNLLForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
                                                                                bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    __global__ void BNLLBackward(const int n, const Dtype* in_diff, const Dtype* in_data, Dtype* out_diff){
        CUDA_KERNEL_LOOP(index, n){
            Dtype expval = exp(min(in_data[index], Dtype(kBNLL_THRESHOLD)));
            out_diff[index] = in_diff[index] * expval / (expval + 1.);
        }
    }
    
    template <typename Dtype>
    void BNLLLayer<Datype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* bottom_data = bottom[0]->gpu_data();
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            
            BNLLBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, 														bottom_data, bottom_diff);
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(BNLLLayer);
}
```

1. 其中`CUDA_POST_KERNEL_CHECK`定义如下：

   ```c++
   // CUDA: various checks for different function calls.
   #define CUDA_CHECK(condition) \
     /* Code block avoids redefinition of cudaError_t error */ \
     do { \
       cudaError_t error = condition; \
       CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
     } while (0)
         
   // CUDA: check for error after kernel execution and exit loudly if there is one.
   #define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
   ```
