# neuron_layer Layer
neuron layer在neuron_layer.hpp, neuron_layer.cpp中实现

**neuron_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class NeuronLayer: public Layer<Dtype>{
   	public:
        explicit NeuronLayer(const LayerParameter& param)：
            Layer<Dtype>(param){}
        
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline int ExactNumBottomBlobs() const {return 1;}
        virtual inline int ExactNumTopBlobs() const {retun 1;}
    };
}
```

**neuron_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top){
        top[0]->ReshapeLike(*bottom[0]);
    }
    
    INSTANTIATE_CLASS(NeuronLayer);
}
```



