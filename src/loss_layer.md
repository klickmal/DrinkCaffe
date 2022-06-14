# Loss Layer
loss layer在loss_layer.hpp和loss_layer.cpp中实现

**loss_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    const float kLOG_THRESHOLD = 1e-20;
    
    // 损失层有两个输入 blob，分别存储 prediction 和 ground-truth 标签，输出是一个单一的 blob 来表示 loss
	 // 损失层一般只对 prediction blob 进行反向传播 
    
    template <typename Dtype>
    class LossLayer : public Layer<Dtype>{
    public:
        explicit LossLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        // 输入一般为 2 个 blob
        virtual inline int ExactNumBottomBlobs() const {return 2;}
        
        // 为了便于反向传播，建议 Net 设置自动为损失层分配一个 blob，其中该 blob 存储损失值
		// （即使用户没有在 prototxt 明确给出输出 blob）
        virtual inline bool AutoTopBlobs() const {return true;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        
        // 通常我们不能对标签进行反向传播，所以我们不对标签进行前置反向传播
        virtual inline bool AllowForceBackward(const int bottom_index) const{
            return bottom_index != 1;
        }
    };
    
}
```

**loss_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/loss_layer.hpp"

namespace caffe{
    
	template <typename Dtype>
    void LossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
        if(this->layer_param_.loss_weight_size() == 0){
            // LossLayers have a non-zero (1) loss by default.
	   		// 如果没有设置 loss 权重，我们默认 loss 权重为 1
            this->layer_param_.add_loss_weight(Dtype(1));
        }
    }
    
    template <typename Dtype>
    void LossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top){
        if(bottom_size() >= 2){
            CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
                << "The data and label should have the same first dimension.";
        }
        
        // loss 值就是一个标量，所以维度为 0
        vector<int> loss_shape(0);
        top[0]->Reshape(loss_shape);
    }
    
    INSTANTIATE_CLASS(LossLayer);
}

```

