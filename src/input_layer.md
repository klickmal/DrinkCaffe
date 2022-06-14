# Input Layer
Input layer给Net提供数据。

input_layer在源文件input_layer.hpp, input_layer.cpp中实现

**input_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    template <typename Dtype>
	class InputLayer: public Layer<Dtype>{
    public:
        explicit InputLayer(const LayerParameter& param):
            Layer<Dtype>(param){}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        
    	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top){}
    
        virtual inline const char* type() const{return "Input";}
    	virtual inline int ExactNumBottomBlobs() const {return 0;}
    	virtual inline int MinTopBlobs() const {return 1;}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){}

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
								  const vector<bool>& propagate_down,
                                  const vector<Blob<Dtype*>& top){}
    };
}
```

**input_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/input_layer.hpp"

namespace caffe{
    
    // This layer produces N >= 1 top blob(s) to be assigned manually.
  	// Define N shapes to set a shape for each top.
  	// Define 1 shape to set the same shape for every top.
  	// Define no shape to defer to reshaping manually.
    
    template <typename Dtype>
    void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
    {
		const int num_top = top.size();
        const InputParamter& param = this->layer_param_.input_param();
        const int num_shape = param.shape_size();
        
        CHECK(num_shape==0 || num_shape==1 || num_shape==num_top)
            << "Must specify 'shape' once, once per top blob, or not at all: "
            << num_top << " top vs. " << num_shape << " shapes. ";
        if(num_shape > 0){
            for(int i = 0; i < num_top; ++i){
                const int shape_index = (param.shape_size()==1)?0:i;
                top[i]->Reshape(param.shape(shape_index));
            }
        }
    }
    
    INSTANTIATE_CLASS(InputLayer);
    REGISTER_LAYER_CLASS(Input);
}

```

