# Noise Layer
noise layer在noise_layer.hpp, noise_layer.cpp和noise_layer.cu中实现

**noise_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    class NoiseLayer: public NeuronLayer<Dtype>{
    public:
        explict NoiseLayer(const LayerParameter& param)
            :NeuronLayer<Dtype>(param){}
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Noise";}
    
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
        Blob<Dtype> mask;
    };
  
}
```

**noise_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top){
        mask.ReshapeLike(*bottom[0]);
    }
    
    template <typename Dtype>
    void NoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        const int count = bottom[0]->count();
        
        if(this->phase_==TRAIN){
            Dtype data_magnitude = sqrt(bottom[0]->sumsq_data()/Dtype(bottom[0]->count()));
            
            if(this->layer_param_.noise_param().has_gaussian_std()){
                caffe_rng_gaussian<Dtype>(count, this->layer_param_.noise_param().bias(),
                                          data_magnitude*this->layer_param_.noise_param().gaussian_std(),
                                          mask.mutable_cpu_data());
            }
            else if(this->layer_param_.noise_param().has_uniform_range()){
                caffe_rng_uniform<Dtype>(count, this->layer_param_.noise_param().bias() - 
                                         this->layer_param_.noise_param().uniform_range(),
                                        this->layer_param_.noise_param().bias() + 
                                        this->layer_param.noise_param().uniform_range(),
                                        mask.mutable_cpu_data());
            }
            caffe_add(count, bottom, mask.cpu_data(), top_data());
        }
        else{
            caffe_copy(count, bottom_data, top_data);
        }
    }
    
    template <typename Dtype>
    void NoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            bottom[0]->ShareDiff(*top[0]);
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(NoiseLayer);
#endif
    
    INSTANTIATE_CLASS(NoiseLayer);
    REGISTER_LAYER_CLASS(Noise);
}
```

**noise_layer.cu**

```c++
#include <vector>

#include "caffe/layers/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       conbst vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gou_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        const int count = bottom[0]->count();
        
        if(this->phase_ == TRAIN){
            Dtype data_magnitude = sqrt(bottom[0]->sumsq_data()/Dtype(bottom[0]->count()));
            if(this->layer_param_.noise_param().has_gaussian_std()){
                caffe_gpu_rng_gaussian<Dtype>(count, (Dtype)this->layer_param_.noise_param().bias(),
                                data_magnitude * (Dtype)this->layer_param_.noise_param().gaussian_std(),
                                mask.mutable_gpu_data());
            }
            else if(this->layer_param_.noise_param().has_uniform_range()){
                caffe_gpu_rng_uniform<Dtype>(count, (Dtype)this->layer_param_.noise_param().bias()-
                                             (Dtype)this->layer_param_.noise_param().uniform_range(),
                                             (Dtype)this->layer_param_.noise_param().bias() +
                                             (Dtype)this->layer_param_.noise_param().uniform_range(),\
                                             mask.mutable_gpu_data());
            }
            caffe_gpu_add(count, bottom_data, mask.gpu_data(), top_data);
        }
        else{
            caffe_copy(count, bottom_data, top_data);
        }
    }

	template <typename Dtype>
    void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            bottom[0]->ShareDiff(*top[0]);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);
}
```

