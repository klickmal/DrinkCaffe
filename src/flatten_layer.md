# Flatten Layer
flatten_layer是单纯的维度变换, 把 (N,C,H,W) 结构搞成 (N,CHW,1,1) 结构。flatten_layer在源文件flatten_layer.hpp和flatten_layer.cpp中实现

**flatten_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
template <typename>
class FlattenLayer: public Layer<Dtype>{
    public:
        explicit FlattenLayer(const LayerParameter& param):
            Layer<Dtype>(param){}

       virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                            const vector<Blob<Dtype>*>& top);
       
       virtual inline const char* type() const{return "Flatten";}
       virtual inline int ExactNumBottomBlobs() const {return 1;}
       virtual inline int ExactNumTopBlobs() const {return 1;}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype*>& top);
    };
}
```

**flatten_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/flatten_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    void FlattenLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                     const vector<Blob<Dtype>*>& top){
        CHECK_NE(top[0], bottom[0]) << this->type() << "Layer does not"
            "allow in-place computation.";
        
        const int start_axis = bottom[0]->CanonicalAxisAIndex(this->layer_param_.flatten_param().axis());
        const int end_axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.flatten_param().end_axis());
        
        vector<int> top_shape;
        for(int i = 0; i < start_axis; ++i){
            top_shape.push_back(bottom[0]->shape(i));
        }
        
        const int flattened_dim = bottom[0]->count(start_axis, end_axis + 1);
        top_shape.push_back(flattened_dim);
        
        for(int i = end_axis + 1; i < bottom[0]->num_axes(); ++i){
            top_shape.push_back(bottom[0]->shape(i));
        }
        
        top[0]->Reshape(top_shape);
        CHECK_EQ(top[0]->count(), bottom[0]->count());
    }
    
    template <typename Dtype>
    void FlattenLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
         top[0]->ShareData(*bottom[0]);
    }
	
    template <typename Dtype>
    void FlattenLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& parapagate_down,
                                  const vector<Blob<Dtype>*>& bottom){
          bottom[0]->ShareDiff(*top[0]);
     }

     INSTANTIATE_CLASS(FlattenLayer);
     REGISTER_LAYER_CLASS(Flatten);
}

```
