# Parameter Layer
parameter layer在parameter_layer.hpp, parameter_layer.cpp和parameter_layer.cu中实现

**parameter_layer.hpp**

```c++
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/filler.hpp"

namespace caffe{

    template <typename Dtype>
    class ParameterLayer: public Layer<Dtype>{
    public:
        explicit ParameterLayer(const LayerParameter& param)
            :Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top){
            if(this->blobs_.size() > 0){
                LOG(INFO) << "Skipping parameter initialization";
            }
            else{
                this->blobs_.resize(1);
                this->blobs_[0].reset(new Blob<Dtype>());
                this->blobs_[0]->Reshape(this->layer_param_.parameter_param().shape());
                share_ptr<Filler<Dtype>> filler(
                    GetFiller<Dtype>(this>layer_param_.parameter_param().blob_filler()));
                filler->Fill(this->blobs_[0].get());
            }
            top[0]->Reshape(this->layer_param_.parameter_param().shape());
        }
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top){}
        virtual inline const char* type() const {return "Parameter";}
        virtual inline int ExactNumBottomBlobs() const {return 0;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
            top[0]->ShareData(*(this->blobs_[0]));
        }
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom){
            if(this->param_propagate_down_[0]){
                this->blobs_[0]->ShareDiff(*top[0]);
            }
        }
        
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
            top[0]->ShareData(*(this->blobs_[0]));
        }
        
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom){
            if(this->param_propagate_down_[0]){
                this->blobs_[0]->ShareDiff(*top[0]);
            }
        }
    };
}
```

**parameter_layer.cpp**

```c++
#include "caffe/layers/parameter_layer.hpp"

namespace caffe{
   
    INSTANTIATE_CLASS(ParameterLayer);
    REGISTER_LAYER_CLASS(Parameter);
}

```

**parameter_layer.cu**

```c++
#include "caffe/layers/parameter_layer.hpp"

namespace caffe{
    INSTANTIATE_LAYER_GPU_FUNCS(ParameterLayer);
}
```

