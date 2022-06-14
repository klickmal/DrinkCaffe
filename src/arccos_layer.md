# Arccos Layer
Arccosine layer在源文件arccos_layer.hpp, arccos_layer.cpp和arccos_layer.cu中实现。arccos层计算公式如下：
$$
y = arccos(x)
$$
arccos层求导：
$$
y = arccos(x) \Rightarrow cos(y) = x
 \Rightarrow -sin(y)y^{'}=1 \\ \Rightarrow y^{'}=\frac{-1}{sin(y)}\Rightarrow \frac{-1}{\sqrt{1-cos(y)^{2} } } = \frac{-1}{\sqrt{1-x^{2} } }    
$$
**arccos_layer.hpp**

```c++
namespace caffe{
    
template <typename Dtype>
    class ArcosLayer: public NeuronLayer<Dtype>{
    public:
        explicit ArccosLayer(const LayerParameter& param):
            NeuronLayer<Dtype>(param){}
        virtual inline const char* type() const{return "Arccos"};
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

**arccos_layer.cpp**

```c++
namespace caffe{
    
    template <typename Dtype>
    void ArcosLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            Dtype* top_data = top[0]->mutable_cpu_data();
            const int count = bottom[0]->count();
            for(int i=0; i < count; i++){
            	top_data[i] = acosf(bottom_data[i]);
            }
        }

    void ArccosLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<bool>& parapagate_down,
                                  const vector<Blob<Dtype>*>& top){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            const Dtype* top_diff = top[0]->cpu_diff();
            const int count = bottom[0]->count();

            if(propagate_down[0]){
                Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
                for(int i=0; i<count; i++)
                {
                    Dtype fixed_in_data = std::min(bottom_data[i], Dtype(1.0)-Dtype(0.01));
                    bottom_diff[i] = -1*top_diff[i]*/sqrtf(1.0 - fixed_in_data*fixed_in_data);
                }
            }
        }

    #ifdef CPU_ONLY
        STUB_GPU(ArccosLayer);
    #endif
    
    INSTANTIATE_CLASS(ArccosLayer);
    REGISTER_LAYER_CLASS(Arccos);
}

```

**arccos_layer.cu**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/arccos_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void ArccosForward(const int n, const Dtype* in, Dtype* out){
        CUDA_KERNEL_LOOP(index, n){
            Dtype fixed_in_data = min(in[index], Dtype(1.0)-Dtype(1e-4));
            fixes_in_data = max(fixed_in_data, Dtype(-1.0)+Dtype(1e-4));
            out[index] = acosf(fixed_in_data);
        }
    }
    
    template <typename Dtype>
    __global__ void ArccosBackward(const int n, const Dtype* in_diff, 
                                   const Dtype* in_data, Dtype* out_diff){
        CUDA_KERNEL_LOOP(index, n){
            Dtype fixed_in_data = min(in_data[index], Dtype(1.0)-Dtype(1e-4));
            fixed_in_data = max(fixed_in_data, Dtype(-1) + Dtype(1e-4));
            out_diff[index] = in_diff[index] * 
                				-1 / sqrt(1.0f - fixed_in_data * fixed_in_data);
        }
    }
    
    template <typename Dtype>
    void ArccosLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();
        
        ArccosForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            			count, bottom_data, top_data);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template <typename Dtype>
    void ArccosLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        const int count = bottom[0]->count();
        
        if(propagate_down[0]){
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            ArccosBackward<Dtype><<< CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            						count, top_diff, bottom_data, bottom_diff);
            CUDA_POST_KERNEL_CHECK;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(ArccosLayer);
}
```

