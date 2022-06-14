# Layer
Layer在layer.hpp和layer.cpp中实现
**layer.hpp**

```c++
// LayerParameter 定义如下：
//message LayerParameter {
//  optional string name = 1; // the layer name: Layer 名称
//  optional string type = 2; // the layer type: Layer type
//  repeated string bottom = 3; // the name of each bottom blob: bottom blob入参名称
//  repeated string top = 4; // the name of each top blob：top blob出参名称 // The train / test phase for computation.
//  optional Phase phase = 10; //是用于模型训练还是测试，TRAIN值为0：用于训练，TEST值为1：用于测试 // The amount of weight to assign each top blob in the objective.
//  // Each layer assigns a default value, usually of either 0 or 1,
//  // to each top blob.
//  repeated float loss_weight = 5;// 每个输出top blob的loss权重 // Specifies training parameters (multipliers on global learning constants,
//  // and the name and other settings used for weight sharing).
//  repeated ParamSpec param = 6; //特定训练参数，可以查看ParamSpec结构 // The blobs containing the numeric parameters of the layer.
//  repeated BlobProto blobs = 7; // 每层的参数 blob // Specifies whether to backpropagate to each bottom. If unspecified,
//  // Caffe will automatically infer whether each input needs backpropagation
//  // to compute parameter gradients. If set to true for some inputs,
//  // backpropagation to those inputs is forced; if set false for some inputs,
//  // backpropagation to those inputs is skipped.
//  //
//  // The size must be either 0 or equal to the number of bottoms.
//  repeated bool propagate_down = 11; // Rules controlling whether and when a layer is included in the network,
//  // based on the current NetState.  You may specify a non-zero number of rules
//  // to include OR exclude, but not both.  If no include or exclude rules are
//  // specified, the layer is always included.  If the current NetState meets
//  // ANY (i.e., one or more) of the specified rules, the layer is
//  // included/excluded.
//  repeated NetStateRule include = 8;
//  repeated NetStateRule exclude = 9; // Parameters for data pre-processing.
//  optional TransformationParameter transform_param = 100; // Parameters shared by loss layers.
//  optional LossParameter loss_param = 101; // Layer type-specific parameters.
//  //
//  // Note: certain layers may have more than one computational engine
//  // for their implementation. These layers include an Engine type and
//  // engine parameter for selecting the implementation.
//  // The default for the engine is set by the ENGINE switch at compile-time.
//  optional AccuracyParameter accuracy_param = 102;
//  optional ArgMaxParameter argmax_param = 103;
//  optional BatchNormParameter batch_norm_param = 139;
//  optional BiasParameter bias_param = 141;
//  optional ClipParameter clip_param = 148;
//  optional ConcatParameter concat_param = 104;
//  optional ContrastiveLossParameter contrastive_loss_param = 105;
//  optional ConvolutionParameter convolution_param = 106;
//  optional CropParameter crop_param = 144;
//  optional DataParameter data_param = 107;
//  optional DropoutParameter dropout_param = 108;
//  optional DummyDataParameter dummy_data_param = 109;
//  optional EltwiseParameter eltwise_param = 110;
//  optional ELUParameter elu_param = 140;
//  optional EmbedParameter embed_param = 137;
//  optional ExpParameter exp_param = 111;
//  optional FlattenParameter flatten_param = 135;
//  optional HDF5DataParameter hdf5_data_param = 112;
//  optional HDF5OutputParameter hdf5_output_param = 113;
//  optional HingeLossParameter hinge_loss_param = 114;
//  optional ImageDataParameter image_data_param = 115;
//  optional InfogainLossParameter infogain_loss_param = 116;
//  optional InnerProductParameter inner_product_param = 117;
//  optional InputParameter input_param = 143;
//  optional LogParameter log_param = 134;
//  optional LRNParameter lrn_param = 118;
//  optional MemoryDataParameter memory_data_param = 119;
//  optional MVNParameter mvn_param = 120;
//  optional ParameterParameter parameter_param = 145;
//  optional PoolingParameter pooling_param = 121;
//  optional PowerParameter power_param = 122;
//  optional PReLUParameter prelu_param = 131;
//  optional PythonParameter python_param = 130;
//  optional RecurrentParameter recurrent_param = 146;
//  optional ReductionParameter reduction_param = 136;
//  optional ReLUParameter relu_param = 123;
//  optional ReshapeParameter reshape_param = 133;
//  optional ScaleParameter scale_param = 142;
//  optional SigmoidParameter sigmoid_param = 124;
//  optional SoftmaxParameter softmax_param = 125;
//  optional SPPParameter spp_param = 132;
//  optional SliceParameter slice_param = 126;
//  optional SwishParameter swish_param = 147;
//  optional TanHParameter tanh_param = 127;
//  optional ThresholdParameter threshold_param = 128;
//  optional TileParameter tile_param = 138;
//  optional WindowDataParameter window_data_param = 129;
//}

```



```c++
#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

namespace boost {class mutex;} `	

// layer.hpp: 父类Layer，定义所有layer的基本接口。
// data_layers.hpp: 继承自父类Layer，定义与输入数据操作相关的子Layer，例如DataLayer，HDF5DataLayer和ImageDataLayer等。
// vision_layers.hpp: 继承自父类Layer，定义与特征表达相关的子Layer，例如ConvolutionLayer，PoolingLayer和LRNLayer等。
// neuron_layers.hpp: 继承自父类Layer，定义与非线性变换相关的子Layer，例如ReLULayer，TanHLayer和SigmoidLayer等。
// loss_layers.hpp: 继承自父类Layer，定义与输出误差计算相关的子Layer，例如EuclideanLossLayer，SoftmaxWithLossLayer和HingeLossLayer等。
// common_layers.hpp: 继承自父类Layer，定义与中间结果数据变形、逐元素操作相关的子Layer，例如ConcatLayer，InnerProductLayer和SoftmaxLayer等。
// layer_factory.hpp: Layer工厂模式类，负责维护现有可用layer和相应layer构造方法的映射表。
// 每个Layer根据自身需求的不同，会定义CPU或GPU版本的实现，例如ConvolutionLayer的CPU和GPU实现就定义在了两个文件中conv_layer.cpp, conv_layer.cu.
    
namespace caffe{
    
    template <typename Dtype>
    class Layer{
    public:
        explicit Layer(const LayerParameter& param)
            : layer_param_(param){
                phase_ = param.phase();
                if(layer_param_.blobs_size() > 0){
                    blobs_.resize(layer_param_.blob_size());
                    for(int i=0; i<layer_param_.blobs_size(); ++i){
                        // 必须先reset一下,不然会报错,为空的,然后再反序列化拷贝
						// 如果没有reset,会报/.. Assertion `px != 0' failed.错误
                        blobs_[i].reset(new Blob<Dtype>());
                        blobs_[i]->FromProto(layer_param_.blobs(i));
                    }
                }
            }
        virtual ~Layer() {}
        
        void SetUp(const vector<Blob<Dtype>*>& bottom,
                  const vector<Blob<Dtype>*>& top){
            CheckBlobCounts(bottom, top);
            LayerSetUp(bottom, top);
            Reshape(bottom, top);
            SetLossWeights(top); // ????
        }
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {}
        
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top) = 0;
        
        // 给定bottom blobs, 计算top blobs并返回loss值
        inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        // 给定top blobs梯度，计算bottom blobs的梯度
        inline void Backward(const vector<Blob<Dtype>*> top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
        
        // Set phase: TRAIN or TEST
        inline void SetPhase(Phase p){
            phase_ = p;
        }
        
        // Returns the vector of learnable parameter blobs.
        vector<shared_ptr<Blob<Dtype>>>& blobs(){
            return blobs_;
        }
        
        void SetBlobs(const vector<Blob<Dtype>*>& weights){
            CHECK_EQ(blobs_.size(), weights.size());
            for(int i=0; i<weights.size(); ++i)
                blobs_[i].reset(weights[i]);
        }
        
        // Gets blobs with normal pointer
        vector<Blob<Dtype>*> GetBlobs(){
            vector<Blob<Dtype>*> ans;
            for(int i=0; i<blobs_.size(); ++i)
                ans.push_back(blobs_[i].get());
            return ans;
        }
        
        // Returns the layer parameter.
        const LayerParameter& layer_param() const {return layer_param_;}
        
        // Writes the layer parameter to a protocol buffers
        virtual void ToProto(LayerParameter* param, bool write_diff=false);
        
        // Returns the scalar loss associated with a top blob at a given index.
        inline Dtype loss(const int top_index) const{
            return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
        }
        
        // Sets the loss associated with a top blob at a given index.
        inline void set_loss(const int top_index, const Dtype value){
            if(loss_.size() <= top_index){
                loss_.resize(top_index+1, Dtype(0));
            }
            loss_[top_index] = value;
        }
        
        virtual inline const char* type() const {return "";}
        
        virtual inline int ExactNumBottomBlobs() const {return -1;}
        
        virtual inline int MinBottomBlobs() const {return -1;}
        
        virtual inline int MaxBottomBlobs() const {return -1;}
        
        virtual inline int ExactNumTopBlobs() const {return -1;}
        
        virtual inline int MinTopBlobs() const {return -1;}
        
        virtual inline int MaxTopBlobs() const {return -1;}
        
        virtual inline bool EqualNumBottomTopBlobs() const {return false;}
        
        // Return whether "anonymous" top blobs are created automatically by the layer.
		// If this method returns true, Net::Init will create enough "anonymous" top
		// blobs to fulfill the requirement specified by ExactNumTopBlobs() or MinTopBlobs().
        virtual inline bool AutoTopBlobs() const {return false;}
        
        // Return whether to allow force_backward for a given bottom blob index.
		// If AllowForceBackward(i) == false, we will ignore the force_backward
		// setting and backpropagate to blob i only if it needs gradient information
		// (as is done when force_backward == false).
        virtual inline bool AllowForceBackward(const int bottom_index) const{
            return true;
        }
        
        inline bool param_propagate_down(const int param_id){
            return (param_propagate_down_.size() > param_id) ?
                param_propagate_down[param_id] : false;
        }
        
        inline void set_param_propagate_down(const int param_id, const bool value){
            if(param_propagate_down_.size() <= param_id){
                param_propagate_down_.resize(param_id+1, true);
            }
            param_propagate_down_[param_id] = value;
        }
        
    protected:
        //protobuf文件中存储的layer参数,从protocal buffers格式的网络结构说明文件中读取
        LayerParameter layer_param_;
        //层状态，参与网络的训练还是测试
        Phase phase_;
        // 储存可学习参数层权值和偏置参数
        vector<shared_ptr<Blob<Dtype>>> blobs_;
        // 标志每个可学习参数blob是否需要计算反向传递的梯度值
        vector<bool> param_propagate_down_;
        // The vector that indicates whether each top blob has a non-zero weight in
		// the objective function
        vector<Dtype> loss_;
        
        // Using the CPU device, compute the layer output. 
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top) = 0;
        // Using the GPU device, compute the layer output. Fall back to Forward_cpu() if unavailable.
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
            return Forward_cpu(bottom, top);
        }
        // Using the CPU device, compute the gradients for any parameters and
		// for the bottom blobs if propagate_down is true.
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom) = 0;
        // Using the GPU device, compute the gradients for any parameters and
		// for the bottom blobs if propagate_down is true.
		// Fall back to Backward_cpu() if unavailable.
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom){
            Backward_cpu(top, propagate_down, bottom);
        }
        
        // Called by the parent Layer's SetUp to check that the number of bottom
		// and top Blobs provided as input match the expected numbers specified by
		// the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
        virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top){
            if(ExactNumBottomBlobs() >= 0){
                CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
                    << type() << " Layer takes " << ExactNumBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if(MinBottomBlobs() >= 0){
                CHECK_LE(MinBottomBlobs(), bottom.size())
                    << type() << " Layer takes at least " << MinBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if(MaxBottomBlobs() >= 0){
                CHECK_GE(MaxBottomBlobs(), bottom.size())
                    << type() << " Layer takes at most " << MaxBottomBlobs()
                    << " bottom blob(s) as input.";
            }
            if(ExactNumTopBlobs() >=0){
                CHECK_EQ(ExactNumTopBlobs(), top.size())
                    << type() << " Layer produces " << ExactNumTopBlobs()
                    << "top blob(s) as output.";
            }
            if(MinTopBlobs() >= 0){
                CHECK_LE(MinTopBlobs(), top.size())
                    << type() << " Layer produces at least " << MinTopBlobs() 
                    << " top blob(s) as output.";
            }
            if(MaxTopBlobs() >= 0){
                CHECK_GE(MaxTopBlobs(), top.size())
                    << type() << " Layer produces at most " << MaxTopBlobs()
                    << " top Blob(s) as output.";
            }
            if(EqualNumBottomTopBlobs())
            {
                CHECK_EQ(bottom.size(), top.size())
                    << type() << " Layer produces one top blob as output for each "
                    << "bottom blob input.";
            }
        }
        
        // Called by SetUp to initialize the weights associated with any top blobs in
		// the loss function. Store non-zero loss weights in the diff blob.
        inline void SetLossWeights(const vector<Blob<Dtype>*>& top){
            const int num_loss_weights = layer_param_.loss_weight_size();
            if(num_loss_weights){
                CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
                    "unspecified or specified once per top blob.";
                for(int top_id=0; top_id<top.size(); ++top_id){
                    const Dtype loss_weight = layer_param_.loss_weight(top_id);
                    if(loss_weight == Dtype(0)) {continue;}
                    this->set_loss(top_id, loss_weight);
                    
                    const int count = top[top_id]->count();
                    Dtype* loss_multiplier = top[top_id]->mutbale_cpu_diff();
                    caffe_set(count, loss_weight, loss_multiplier);
                }
            }
        }
        
    private:
        DISABLE_COPY_AND_ASSIGN(Layer);
    };
    
    template <typename Dtype>
    inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        Dtype loss = 0;
        Reshape(bottom, top);
        switch(Caffe::mode()){
            case Caffe::CPU:
                Forward_cpu(bottom, top);
                for(int top_id=0; top_id<top.size(); ++top_id){
                    if(!this->loss(top_id)) {continue;}
                    const int count = top[top_id]->count();
                    const Dtype* data = top[top_id]->cpu_data();
                    const Dtype* loss_weights = top[top_id]->cpu_diff();
                    loss += caffe_cpu_dot(count, data, loss_weights);
                }
                break;
            case Caffe::GPU:
                Forward_gpu(bottom, top);
#ifndef CPU_ONLY
                for(int top_id=0; top_id<top.size(); ++top_id){
                    if(!this->loss(top_id)) {continue;}
                    const int count = top[top_id]->count();
                    const Dtype* data = top[top_id]->gpu_data();
                    const Dtype* loss_weights = top[top_id]->gpu_diff();
                    Dtype blob_loss = 0;
                    caffe_gpu_dot(count, data, loss_weights, &blob_loss);
                    loss += blob_loss;
                }
#endif
                break;
            default:
                LOG(FATAL) << "Unknown caffe mode.";
        }
        return loss;
    }
    
    template <typename Dtype>
    inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom){
        switch(Caffe::mode()){
            case Caffe::CPU:
                Backward_cpu(top, propagate_down, bottom);
                break;
            case CAFFE::GPU:
                Backward_gpu(top, propagate_down, bottom);
                break;
            default:
                LOG(FATAL) << "Unknown caffe mode.";
        }
    }
    
    // Serialize LayerParameter to protocol buffer
    template <typename Dtype>
    void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff){
        param->Clear();
        param->CopyFrom(layer_param_);
        param->clear_blobs();
        for(int i=0; i<blobs_.size(); ++i){
            blobs_[i]->ToProto(param->add_blobs(), write_diff);
        }
    }
	
}
```

**layer.cpp**

```c++
#include "caffe/layer.hpp"

namespace caffe{
    INSTANTIATE_CLASS(Layer);
}
```



