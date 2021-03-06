# Custom Layers
custom layer类在custom_layer.hpp中实现。
**custom_layers.hpp**

```c++
#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class InsanityLayer: public NeuronLayer<Dtype>{
    public:
        explicit InsanityLayer(const LayerParameter& param)
            : NeuronLayer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Insanity";}
        
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
        
        Dtype lb_, ub_, mean_slope;
        Blob<Dtype> alpha;
        Blob<Dtype> bottom_memory_;
    };
    
    template <typename Dtype>
    class ROIPoolingLayer: public Layer<Dtype>{
    public:
        explicit ROIPoolingLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "ROIPooling";}
        
        virtual inline int MinBottomBlobs() const {return 2;}
        virtual inline int MaxBottomBlobs() const {return 2;}
        virtual inline int MinTopBlobs() const {return 1;}
        virtual inline int MaxTopBlobs() const {return 1;}
        
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propogate_down,
                                  const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
        
        int channels_;
      	int height_;
        int width_;
        int pooled_height_;
        int pooled_width_;
        Dtype spatial_scale_;
        Blob<int> max_idx_;
    };
    
    template <typename Dtype>
    class LocalLayer: public Layer<Dtype>{
    public:
        explicit LocalLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Local";}
        virtual inline int MinBottomBlobs() const {return 1;}
        virtual inline int MinTopBlobs() const {return 1;}
        virtual inline bool EqualNumBottomTopBlobs() const {return true;}
    
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
        
        int kernel_size_;
        int stride_;
        int num_;
        int channels_;
        int pad_;
        int height_;
        int width_;
        int height_out_;
        int width_out_;
        int num_output_;
        bool bias_term_;
        
        int M_;
        int K_;
        int N_;
        
        Blob<Dtype> col_buffer_;
    };
    
    template <typename Dtype>
    class SmoothL1LossLayer: public LossLayer<Dtype>{
    public:
        explicit SmoothL1LossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param), diff_() {}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "SmoothL1Loss";}
        virtual inline int ExactNumBottomBlobs() const {return -1;}
        virtual inline int MinBottomBlobs() const {return 2;}
        virtual inline int MaxBottomBlobs() const {return 3;}
        
        virtual inline bool AllowForceBackward(const int bottom_index) const{
            return true;
        }
     
    protected:
        virtual viod Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual viod Forward_gpu(const vector<Blob<Dtype>*>& bottom,
         const vector<Blob<Dtype>*>& top);
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
        
        Blob<Dtype> diff_;
        Blob<Dtype> errors_;
        bool has_weights_;
        Dtype turn_point_;
    };
    
    template <typename Dtype>
    class TripletLossLayer: public LossLayer<Dtype>{
    public:
        explicit TripletLossLayer(const LayerParameter& param): LossLayer<Dtype>(param){}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual inline int ExactNumBottomBlobs() const {return 3;}
        virtual inline const char* type() const {return "TripletLoss";}
        
        virtual inline bool AllowForceBackward(const int bottom_index) const{
            return bottom_index != 3;
        }
        
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
        
        Blob<Dtype> diff_ap_;  // cached for backward pass
        Blob<Dtype> diff_an_;  // cached for backward pass
        Blob<Dtype> diff_pn_;  // cached for backward pass

        Blob<Dtype> diff_sq_ap_;  // cached for backward pass
        Blob<Dtype> diff_sq_an_;  // tmp storage for gpu forward pass

        Blob<Dtype> dist_sq_ap_;  // cached for backward pass
        Blob<Dtype> dist_sq_an_;  // cached for backward pass

        Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
        Blob<Dtype> dist_binary_;  // tmp storage for gpu forward pass
    };
    
    template <typename Dtype>
    class TransformerLayer: public Layer<Dtype>{
    public:
        explicit TransformerLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Transformer";}
        virtual inline int ExactNumBottomBlobs() const {return 2;}
        virtual inline int MinTopBlobs() const {return 1;}
    
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
        
        int count_;
        Blob<Dtype> CoordinateTarget;
        Blob<Dtype> CoordinateSource;
        Blob<Dtype> InterpolateWeight;
    };
    
    template <typename Dtype>
    class GramLayer: public Layer<Dtype>{
    public:
      explicit GramLayer(const LayerParameter& param)
          : Layer<Dtype>(param) {}
       
        virtual inline const char* type() const {return "Gram";}
        virtual inline int ExactNumBottomBlobs() const {return 1;}
        virtual inline int  MinTopBlobs() const {return 1;}
     
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
   	
    template <typename Dtype>
    class EltwiseAffineLayer: public NeuronLayer<Dtype>{
    public:
        explicit EltwiseAffineLayer(const LayerParameter& param)
            : NeuronLayer<Dtype>(param) {}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "EltwiseAffine";}
     
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
        
        bool channel_shared_;
        Blob<Dtype> multiplier_;
        Blob<Dtype> bias_multiplier_;
        Blob<Dtype> backward_buff_;
        Blob<Dtype> bottom_memory_;
    };
    
    // Get the sub-region features around some specific points
    template <typename Dtype>
    class SubRegionLayer: public Layer<Dtype>{
    public:
    	explicit SubRegionLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "SunRegion";}
        virtual inline int ExactNumBottomBlobs() const {return 3;}
        virtual inline int MinTopBlobs() const {return 1;}
        
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
        
        int height_;
        int width_;
        int data_height_;
        int data_width_;
        int as_dim_;
    };
    
    
    template <typename Dtype>
    class NoiseLayer: public NeuronLayer<Dtype>{
    public:
        explicit NoiseLayer(const LayerParameter& param)
            : NeuronLayer<Dtype>(param) {}
        
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
    };
}
```



