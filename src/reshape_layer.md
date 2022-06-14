# Reshape Layer
reshape layer在reshape_layer.hpp, reshape_layer.cpp中实现

**reshape_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class ReshapeLayer: public Layer<Dtype>{
    public:
        explicit ReshapeLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Reshape";}
        virtual inline int ExactNumBottomBlobs() const {return 1;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){}
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){}
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom){}
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom){}
        
        vector<int> copy_axes_; 
        int referred_axis_;
        int constant_count_;  
    };
}
```

**reshape_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/reshape_layer.hpp"

namespace caffe{
    
	template <typename Dtype>
    void ReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        CHECK_NE(top[0], bottom[0]) << this->type() << "Layer does not "
            "allow in-place computation.";
        inferred_axis_ = -1;
        copy_axes_.clear();
        
        
        const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
        const int top_num_axes = top_blob_shape.dim_size();
        constant_count_ = 1;
        
        for(int i = 0; i < top_num_axes; ++i){
            const int top_dim = top_blob_shape.dim(i);
            
            if(top_dim == 0){
                copy_axes_.push_back(i);
            }
            else if(top_dim == -1){
                CHECK_EQ(inferred_axis_, -1) << "new shape contains mutiple "
                    << "-1 dims; at most a single (1) value of -1 may be specified";
                inferred_axis_ = i;
            }
            else{
                constant_count_ *= top_dim；
            }
        }
    }
    
    template <typename Dtype>
    void ReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
        const int input_start_axis = this->layer_param_.reshape_param().axis();
        const int start_axis = (input_start_axis >= 0) ? input_start_axis : 
        					bottom[0]->num_axes() + input_start_axis + 1;
        CHECK_GE(start_axis, 0) << "axis" << input_start_axis << " out of range";
        CHECK_LE(start_axis, bottom[0]->num_axes()) << "axis " << input_start_axis
            << "out of range for " << bottom[0]->num_axes() << "-D input blob";
        
        const int num_axes = this->layer_param_.reshape_param().num_axes();
        CHECK_GE(num_axes, -1) << "num_axes must be >= 0, or -1 for all";
        
        const int end_axes = (num_axes == -1) ? bottom[0]->num_axes() : 
        						(start_axis + num_axes);
        CHECK_LE(end_axis, bottom[0]->num_axes())
            << "end_axis = axis + num_axes is out of range";
        
        const int num_axes_replaced = end_axis - start_axis;
        const int num_axes_retained = bottom[0]->num_axes() - num_axes_replaced;
        const BlobShape& top_blob_shape = this->layer_param_.reshape_param().shape();
        const int num_new_axes = top_blob_shape.dim_size();
        
        vector<int> top_shape(num_axes_retained + num_new_axes);
        int top_shape_index = 0;
        for(int i = 0; i < start_axis; ++i){
            top_shape[top_shape_index++] = bottom[0]->shape(i);
        }
        
        for(int i = 0; i < num_new_axes; ++i){
            top_shape[top_shape_index++] = top_blob_shape.dim(i);
        }
        
        for(int i = end_axis; i < bottom[0]->num_axes(); ++i){
            top_shape[top_shape_index++] = bottom[0]->shape(i);
        }
        
        CHECK_EQ(top_shape_index, top_shape.size());
        for(int i = 0; i < copy_axes_.size(); ++i){
            const int copy_axis_index = copy_axes_[i];
            CHECK_GT(bottom[0]->num_axes(), start_axis + copy_axis_index)
                << "new shape contains a 0, but there are no corresponding bottom axis "
                << "to copy";
            top_shape[start_axis + copy_axis_index] = 
               	 					bottom[0]->shape(start_axis + copy_axis_index);
        }
        
        if(inferred_axis_ >= 0){
            int explicit_count = constant_count_;
            explicit_count *= bottom[0]->count(0, start_axis);
            explicit_count *= bottom[0]->count(end_axis);
            
            for(int i = 0; i < copy_axes_>size(); ++i){
                const int copy_axis_index = copy_axes_[i];
                explicit_count *= top_shape[start_axis + copy_axis_index];
            }
            
            CHECK_EQ(0, bottom[0]->count() % explicit_count) << "bottom count ("
                << bottom[0]->count() << ") must be divisible by the product of "
                << "the specified dimensions (" << explicit_count << ")";
            const int inferred_dim = bottom[0]->count() / explicit_count;
            top_shape[start_axis + inferred_axis_] = inferred_dim;
        }
        top[0]->Reshape(top_shape);
        CHECK_EQ(top[0]->count(), bottom[0]->count())
            << "output count must match input count";
        
        top[0]->ShareData(*bottom[0]);
        top[0]->ShareData(*bottom[0]);
    }
    
    INSTANTIATE_CLASS(ReshapeLayer);
    REGISTER_LAYER_CLASS(Reshape);
}

```

 1. 参数解读

    ```c++
    layer {
            name: "reshape"
            type: "Reshape"
            bottom: "input"
            top: "output"
            reshape_param {
           	shape {
               	dim: 0  # copy the dimension from below
                dim: 2
                dim: 3
                dim: -1 # infer it from the other dimensions
              	}
            }
          }
    
    #有一个可选的参数组shape, 用于指定blob数据的各维的值（blob是一个四维的数据：n*c*w*h）。  
    #dim:0  表示维度不变，即输入和输出是相同的维度。  
    #dim:2 或 dim:3 将原来的维度变成2或3  
    #dim:-1 表示由系统自动计算维度。数据的总量不变，系统会根据blob数据的其它三维来自动计算当前维的维度值 。  
    
    #假设原数据为：32*3*28*28， 表示32张3通道的28*28的彩色图片  
    #   shape {  
    #   dim: 0  32-32  
    #   dim: 0  3-3  
    #   dim: 14 28-14  
    #   dim: -1 #让其推断出此数值  
    #   }  
    
    #输出数据为：32*3*14*56 
    ```

    ```c++
    message ReshapeParameter {
      // Specify the output dimensions. If some of the dimensions are set to 0,
      // the corresponding dimension from the bottom layer is used (unchanged).
      // Exactly one dimension may be set to -1, in which case its value is
      // inferred from the count of the bottom blob and the remaining dimensions.
      // For example, suppose we want to reshape a 2D blob "input" with shape 2 x 8:
      //
      //   layer {
      //     type: "Reshape" bottom: "input" top: "output"
      //     reshape_param { ... }
      //   }
      //
      // If "input" is 2D with shape 2 x 8, then the following reshape_param
      // specifications are all equivalent, producing a 3D blob "output" with shape
      // 2 x 2 x 4:
      //
      //   reshape_param { shape { dim:  2  dim: 2  dim:  4 } }
      //   reshape_param { shape { dim:  0  dim: 2  dim:  4 } }
      //   reshape_param { shape { dim:  0  dim: 2  dim: -1 } }
      //   reshape_param { shape { dim:  0  dim:-1  dim:  4 } }
      //
      optional BlobShape shape = 1;
    
      // axis and num_axes control the portion of the bottom blob's shape that are
      // replaced by (included in) the reshape. By default (axis == 0 and
      // num_axes == -1), the entire bottom blob shape is included in the reshape,
      // and hence the shape field must specify the entire output shape.
      //
      // axis may be non-zero to retain some portion of the beginning of the input
      // shape (and may be negative to index from the end; e.g., -1 to begin the
      // reshape after the last axis, including nothing in the reshape,
      // -2 to include only the last axis, etc.).
      //
      // For example, suppose "input" is a 2D blob with shape 2 x 8.
      // Then the following ReshapeLayer specifications are all equivalent,
      // producing a blob "output" with shape 2 x 2 x 4:
      //
      //   reshape_param { shape { dim: 2  dim: 2  dim: 4 } }
      //   reshape_param { shape { dim: 2  dim: 4 } axis:  1 }
      //   reshape_param { shape { dim: 2  dim: 4 } axis: -3 }
      //
      // num_axes specifies the extent of the reshape.
      // If num_axes >= 0 (and axis >= 0), the reshape will be performed only on
      // input axes in the range [axis, axis+num_axes].
      // num_axes may also be -1, the default, to include all remaining axes
      // (starting from axis).
      //
      // For example, suppose "input" is a 2D blob with shape 2 x 8.
      // Then the following ReshapeLayer specifications are equivalent,
      // producing a blob "output" with shape 1 x 2 x 8.
      //
      //   reshape_param { shape { dim:  1  dim: 2  dim:  8 } }
      //   reshape_param { shape { dim:  1  dim: 2  }  num_axes: 1 }
      //   reshape_param { shape { dim:  1  }  num_axes: 0 }
      //
      // On the other hand, these would produce output blob shape 2 x 1 x 8:
      //
      //   reshape_param { shape { dim: 2  dim: 1  dim: 8  }  }
      //   reshape_param { shape { dim: 1 }  axis: 1  num_axes: 0 }
      //
      // axis是reshape开始的bottom维度, axis若为负数，-1表示最后一个轴的下一个位置，-2表示最后一个轴，跟python numpy不一样
      // nun_axes是reshape维度的数量
      optional int32 axis = 2 [default = 0];
      optional int32 num_axes = 3 [default = -1];
    }
    ```

    
