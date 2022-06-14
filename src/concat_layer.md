# CONCAT Layer

concat layer在源文件**concat_layer.hpp、concat_layer.cpp、concat_layer.cu**中实现。concat layer用来实现两个或多个blob连接，即多输入一输出；支持在batch维度上和channel维度上对blob进行连接。

**concat_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namesapce caffe{
	template <typename Dtype>
    class ConcatLayer: public Layer<Dtype>{
    public: 
      	explicit ConcatLayer(const LayerParameter& param):
        	Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype*>&> top);
        
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom, 
                            const vector<Blob<Dtype>*>& top);
        
      	virtual inline const char* type() const {return "Concat";}
        virtual inline int MinBottomBlobs const {return 1;}
        virtual inline int ExactNumTopBlobs const {return 1;}
     
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
        int num_concats_;
        int concat_input_size_;
        int concat_axis_;
    };
}
```

**concat_layer.cpp**

```c++
#include <vector>
#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{    
    
    template <typename Dtype>
    void ConcatLayer<Dtype>::LayerSetup(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        const ConcatParameter& concat_param = this->layer_param_.concat_param();
        CHECK(!(concat_param.has_axis() && concat_param.has_concat_dim()))
            <<< "Either axis or concat_dim should be specified; not both;"
    }
    
    template <typename Dtype>
    void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top){
        const int num_axes = bottom[0]->num_axes();
        const ConcatParameter& concat_param = this->layer_param_.concat_param();
        
        if(concat_param.has_concat_dim()){
        	concat_axis_ = static_cast<int>(concat_param.concat_dim());
            
            CHECK_GE(concat_axis,0) << "casting concat_dim from uint32 to int32"
				<< "produced negative result; concat_dim must satisfy "
				<< "0 <= concat_dim < " << kMaxBlobAxes;
            CHECK_LT(concat_axis_, num_axes) << "concat_dim out of range.";
        }
        else{
            concat_axis_ = bottom[0]->CanonicalAxisIndex(concat_param.axis());
        }
        
        vector<int> top_shape =  bottom[0]->shape();
        num_concats_ = bottom[0]->count(0, concat_axis_);
        concat_input_size_ = bottom[0]->count(concat_axis_ + 1);
        
        int bottom_count_sum = bottom[0]->count();
        
        for(int i = 0; i < bottom.size(); ++i){
            CHECK_EQ(num_axes, bottom[i]->num_axes()) << "All inputs must have the same *axes.";
            
            for(int j = 0; j < num_axes; ++j){
                if(j==concat_axis_){continue;}
                CHECK_EQ(top_shape[j], bottom[i]->shape(j)) 
                    << "All inputs must have the same shape, except at concat_axis.";
            }
            bottom_count_sum += bottom[i]->count();
            top_shape[concat_axis_] += bottom[i]->shape(concat_axis_);
        }
        
        top[0]->Reshape(top_shape);
        CHECK_EQ(bottom_count_sum, top[0]->count());
        
        if(bottom.size() == 1){
            top[0]->ShareData(*bottom[0]);
            top[0]->ShareDiff(*bottom[0]);
        }
    }
    
    template <typename Dtype>
    void ConcatLayer<Dtype>::Forward_cpu(const vectotr<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
       if(bottom.size() == 1){return;}
       
       Dtype* top_data = top[0]->mutable_cpu_data();
       int offset_concat_axis = 0;
       
       const int top_concat_axis = top[0]->shape(concat_axis_);
       for(int i = 0; i < bottom.size(); ++i){
           const Dtype* bottom_data = bottom[i]->cpu_data();
           const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
           
           for(int n = 0; n < num_concats_; ++n){
               caffe_copy(bottom_concat_axis * concat_input_size, 
                          bottom_data + n*bottom_concat_axis * concat_input_size_,
                          top_data + (n*top_concat_axis+ offset_concat_axis)*concat_input_size_);
           }
           offset_concat_axis += bottom_concat_axis;
       }
    }
    
    template <typename Dtype>
    void ConcatLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
   		
        if(bottom.size() == 1){return;}
        const Dtype* top_diff = top[0]->cpu_diff();
        int offset_concat_axis = 0;
        
        const int top_concat_axis = top[0]->shape(concat_axis_);
        
        for(int i = 0; i < bottom.size(); ++i){
            const int bottom_concat_axis = bottom[i]->shape(concat_axis_};
            if(propagate_down[i]){
                Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
                
                for(int n = 0; n < num_concats_; ++n){
                    caffe_copy(bottom_concat_axis * concat_input_size_, 
                               top_diff + (n*top_concat_axis + offset_concat_axis) * concat_input_size_,
                               bottom_diff + n*bottom_concat_axis * concat_input_size_);
                }
            }
            offset_concat_axis += bottom_concat_axis;
         }
    }
    
    ifdef CPU_ONLY
    STUB_GPU(ConcatLayer);
    #endif
    
    INSTANTIATE_CLASS(ConcatLayer);
    REGISTER_LAYER_CLASS(Concat);
}
```

 1. ConcatParameter定义如下：

    ```c++
    message ConcatParameter {
      // The axis along which to concatenate -- may be negative to index from the
      // end (e.g., -1 for the last axis).  Other axes must have the
      // same dimension for all the bottom blobs.
      // By default, ConcatLayer concatenates blobs along the "channels" axis (1).
      optional int32 axis = 2 [default = 1];
    
      // DEPRECATED: alias for "axis" -- does not support negative indexing.
      optional uint32 concat_dim = 1 [default = 1];
    }
    ```

**concat_layer.cu**

```c++
#include <vector>

#include "caffe/layers/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
   
    template <typename Dtype>
    __global__ void Concat(const int nthreads, const Dtype* in_data, const bool forward, const int 		                         num_concats, const int concat_size, const int top_concat_axis, const int 	                                 bottom_concat_axis, const int offset_concat_axis, Dtype* out_data){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int total_concat_size = concat_size*bottom_concat_size;
            const int concat_num = index / total_concat_size;
            const int concat_index = index % total_concat_size;
            const int top_index= concat_index + 
                				(concat_num * top_concat_axis + offset_concat_axis) * concat_size; 
            if(forward){
                out_data[top_index] = in_data[index];
            }
            else{
                out_data[index] = in_data[top_index];
            }
        }
    }
    
    template <typename Dtype>
    void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                        const vecotr<Blob<Dtype>*>& top){
        if(bottom.size() == 1) {return ;}
        
        Dtype* top_data = top[0]->mutable_gpu_data();
        int offset_concat_axis = 0;
        const int top_concat_axis = top[0]->shape(concat_axis_);
        const bool fForward = true;
        
        for(int i = 0; i < bottom.size(); ++i){
            const Dtype* bottom_data = bottom[i]->gpu_data();
            const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
            const int nthreads = bottom_concat_size * num_concats_;
            
            Concat<Dtype> <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM__THREADS>>>(
                         nthreads, bottom_data, kForward, num_concats_, concat_input_size_, 
                         top_concat_axis, bottom_concat_axis, offset_concat_axis, top_data); 
            offset_concat_axis += bottom_conc at_axis;
        }
    }
    
     template <typename Dtype>
     void ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down, 
                                          const vector<Blob<Dtype>*>& bottom){
         if(bottom.size() == 1) {return;}
         
         const Dtype* top_diff = top[0]->gpu_diff();
         int offset_concat_axis = 0;
         
         const int top_concat_axis = top[0]->shape(concat_axis_);
         const bool kForward = false;
         
         for(int i = 0; i < bottom.size(); ++i){
             const int bottom_concat_axis = bottom[i]->shape(concat_axis_);
             if(propagate_downs[i]){
                 Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
                 const int bottom_concat_size = bottom_concat_axis * concat_input_size_;
                 
                 const int nthreads = bottom_concat_size * num_concats_;
                 Concat<Dtype> <<< CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>> (nthreads, 	                                      top_diff, kForward, num_concats_, concat_input_size_, top_concat_axis, 
                                 bottom_concat_axis, offset _concat_axis, bottom_diff);
             }
             offset_concat_axis += bottom_concat_axis;
         }
     }
    
     INSTANTIATE_LAYER_GPU_FUNCS(ConcatLayer);
}
```

