# Threshold Layer
threshold layer计算公式如下：
$$
Y = 
\begin{cases}
1 & x>threshold \\
0 & x<=threshold \\
\end{cases}
$$


threshold layer在源文件threshold_layer.hpp, threshold_layer.cpp和threshold_layer.cu中实现

**threshold_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
// 阈值层    
template <typename Dtype>
class ThresholdLayer: public NeuronLayer<Dtype>{
    public:
        explicit ThresholdLayer(const LayerParameter& param):
            NeuronLayer<Dtype>(param){}
    	
    	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
    
        virtual inline const char* type() const {return "Threshold";} 

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype*>& bottom){
            NOT_IMPLEMENTED;
        }
    
    	Dtype threshold_;
    };
}
```

**threshold_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    void ThresholdLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top){
        NeuronLayer<Dtype>::LayerSetUp(bottom, top);
        threshold_ = this->layer_param_.threshold_param().threshold();
    }
    
    template <typename Dtype>
    void ThresholdLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
     	const Dtype* bottom_data =  bottom_data[0]->cpu_data();
     	Dtype* top_data = top[0]->mutable_cpu_data();
      	
        const int count = bottom[0]->count();
        for(int i=0; i<count; ++i){
            top_data[i] = (bottom_data[i]>threshold_)?Dtype(1):Dtype(0);
        }
    }

#ifdef CPU_ONLY
	STUP_GPU(ThresholdLayer);
#endif
    
    INSTANTIATE_CLASS(ThresholdLayer);
    REGISTER_LAYER_CLASS(Threshold);
}

```

**threshold_layer.cu**

```c++
#include <vector>

#include "caffe/layers/threshold_layer.hpp"

namespace caffe{

   template <typename Dtype>
   __global__ void ThresholdForward(const int n, const Dtype threshold, 
                                    const Dtype* in, Dtype* out){
       CUDA_KERNEL_LOOP(index, n){
           out[index] = in[index] > threshold ? 1: 0;
       }
   }
   
   template <typenmae Dtype>
   void ThresholdLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->gpu_data();
       Dtype* top_data = top[0]->mutable_gpu_data();
       
       const int count = bottom[0]->count();
       ThresholdForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
       				count, threshold_, bottom_data, top_data);
       CUDA_POST_KERNEL_CHECK;
   }
    
   INSTANTIATE_LAYER_GPU_FORWARD(ThresholdLayer);
}
```

