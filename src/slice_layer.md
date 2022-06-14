# Slice Layer
slice layer层作用是对blob进行切片操作。

slice layer在slice_layer.hpp, slice_layer.cpp和slice_layer.cu中实现。

**slice_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

    // sliceLayer层的作用是对blob块进行切片操作。 该层两个重要的参数是
	// slice_axis和 slice_point(是一个vetor)，分别指明了沿哪一个轴进行切片以及
	// 在该轴上切片的位置
	template <typename Dtype>
	class SliceLayer: public Layer<Dtype>{
    public:
    	explicit SliceLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Slice";}
        virtual inline int ExactNumBottomBlobs() const {return 1;}
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
        
        // 假设bottom[0]的shape={a,b,c,d,e,f}, 并且在c轴(slice_axis_=2)进行切片
        // 则下面的成员变量分别为：
        int count_;			// a*b*c*d*e*f,表示共有多少元素
        int num_slices_;	// a*b
        int slice_size_;	// d*e*f
        int slice_axis_;	// 2
        vector<int> slice_point_; // 给定轴上切片的位置点，例如轴长为6，slice_point_的 值为[1，4，5],
        						  // 则切片的结果分别为：[0,1),[1,4), [4,5), [5,6)
        						  // 总计是4份，相应的top块的数目也是4
    };
}
```

**slice_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
   
    template <typename Dtype>
    void SliceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const SliceParameter& slice_param = this->layer_param_.slice_param();
        CHECK(!(slice_param.has_axis() && slice_param.has_slice_dim()))
            << "Either axis or slice_dim should be specified; not both.";
        
        slice_point_.clear();
        std::copy(slice_param.slice_point().begin(), 
                  slice_param.slice_point().end(),
                  std::back_inserter(slice_point_));
	}
    
    template <typename Dtype>
    void SliceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top){
        const int num_axes = bottom[0]->num_axes();
        const SliceParameter& slice_param = this->layer_param_.slice_param();
        
        // 验证并设置slice_axis_的值, 它表示在哪一个维度或者轴上进行切片。
		// 之前的老版本使用slice_dim参数，新版本使用了axis参数. 这里需要兼顾
		// 一下老版本的参数名
        if(slice_param.has_slice_dim()){
            slice_axis_ = static_cast<int>(slice_param.slice_dim());
            CHECK_GE(slice_axes, 0) << "casting slice_dim from uint32 to int32 "
                << "produced negative result; slice_dim must satisfy "
                << "0 <= slice_dim < " << kMaxBlobAxes;
            CHECK_LT(slice_axis_, num_axes) << "slice_dim out of range. ";
        }
        else
        {
            slice_axis_ = bottom[0]->CanonicalAxisIndex(slice_param.axis());
        }
   		
        vector<int> top_shape = bottom[0]->shape();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
        num_slices_ = bottom[0]->count(0, slice_axis_);
        slice_size_ = bottom[0]->count(slice_axis_ + 1);
        int count = 0;
        
        if(slice_point_.size()){
            CHECK_EQ(slice_point_.size(), top.size() - 1);
            CHECK_LE(top.size(), bottom_slice_axis);
            
            // 假设slice_point_的值为[1，4，5], bottom_slice_axis为6
            // slices为[1,3,1,1]
            int prev = 0;
            vector<int> slices;
            for(int i = 0; i < slice_point_.size(); ++i){
                CHECK_GT(slice_point_[i], prev);
                slices.push_back(slice_point_[i] - prev);
                prev = slice_point_[i];
            }
            // 创建top
            slices.push_back(bottom_slice_axis - prev);
            for(int i = 0; i < top.size(); ++i){
                top_shape[slice_axis_] = slices[i];
                top[i]->Reshape(top_shape);
                count += top[i]->count();
            }
        }
        else
        {
            // 均分blob
            CHECK_EQ(bottom_slice_axis % top.size(), 0)
                << "Number of top blobs (" << top.size() << ") should evenly "
                << "divide input slice axis (" <<  bottom_slice_axis << ")";
            top_shape[slice_axis_] = bottom_slice_axis / top.size();
            for(int i = 0; i < top.size(); ++i){
                top[i]->Reshape(top_shape);
                count += top[i]->count();
            }
        }
        
        CHECK_EQ(count, bottom[0]->count());
        
        // 不做切片操作
        if(top.size() == 1){
            top[0]->ShareData(*bottom[0]);
            top[0]->ShareDiff(*bottom[0]);
        } 
    }
    
    template <typename Dtype>
    void SliceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        if(top.size() == 1) {return;}
        int offset_slice_axis = 0;  // 沿切片轴的偏移
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
        
        for(int i = 0; i < top.size(); ++i){
            Dtype* top_data = top[i]->mutable_cpu_data();
            const int top_slice_axis = top[i]->shape(slice_axis_);
            
            // 假设bottom[0]的shape = [a,b,c,d,e,f], 并且在c轴(对应slice_axis_=2）进行切片，
			// 则下面的成员变量的值分别对应了：
			// int count_;                   // 它的值等于a*b*c*d*e*f，表示共有多少输入元素。
			// int num_slices_;              // 它的值等于a*b
			// int slice_size_;              // 它的值等于d*e*f
			// int slice_axis_;              // 它的值等于2
            for(int n = 0; n < num_slices_; ++n){
                // top_data每次偏移top_slice_axis * slice_size_
                const int top_offset = n * top_slice_axis * slice_size_; 
                // bottom每次偏移量更大
                const int bottom_offset = 
                    (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
                caffe_copy(top_slice_axis * slice_size_,
                          bottom_data + bottom_offset, top_data + top_offset);
            }
            offset_slice_axis += top_slice_axis;
        }
    }
    
    // backforward跟forward过程类似，梯度从top_diff复制到bottom_diff
    template <typename Dtype>
    void SliceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0] || top.size() == 1) {return;}
        int offset_slice_axis = 0;
        
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
        
        for(int i = 0; i < top.size(); ++i){
            const Dtype* top_diff = top[i]->cpu_diff();
            const int top_slice_axis = top[i]->shape(slice_axis_);
            
            for(int n = 0; n < num_slices_; ++n){
                const int top_offset = n * top_slice_axis_ * slice_size_;
                const int bottom_offset = 
                    (n * bottom_slice_axis + offset_slice_axis) * slice_size_;
                caffe_copy(top_slice_axis * slice_size_; 
                          top_diff + top_offset, bottom_diff + bottom_offset);
            }
            offset_slice_axis += top_slice_axis;
        }
    }
    
    
#ifdef CPU_ONLY
    STUB_GPU(SliceLayer);
#endif
    
    INSTANTIATE_CLASS(SliceLayer);
    REGISTER_LAYTER_CLASS(Slice);
    
}

```

**slice_layer.cu**

```c++
#include <vector>

#include "caffe/layers/slice_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    __global__ void Slice(const int nthreads, const Dtype* in_data,
                         const bool forward, const int num_slices, const int slice_size,
                         const int bottom_slice_axis, const int top_slice_axis,
                         const int offset_slice_axis, Dtype* out_data){
        CUDA_KERNEL_LOOP(index, nthreads){
            const int total_slice_size = slice_size * top_slice_axis;
            const int slice_num = index / toptal_slice_size;
            const int slice_index = index % total_slice_size;
            const int bottom_index = slice_index +
                (slice_num * bottom_slice_axis + offset_slice_axis) * slice_size;
            if(forward){
                out_data[index] = in_data[bottom_index];
            }
            else{
                out_data[bottom_index] = in_data[index];
            }
        }
    }
    
    template <typename Dtype>
    void SliceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        if(top.size() == 1) {return;}
        int offset_slice_axis = 0;
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
        const bool kForward = true;
        
        for(int i = 0; i < top.size(); ++i){
            Dtype* top_data = top[i]->mutable_gpu_data();
            const int top_slice_axis = top[i]->shape(slice_axis_);
            const int top_slice_size = top_slice_axis * slice_size_;
            const int nthreads = top_slice_size * num_slices_;
            
            // 并行遍历top[i]中的每一个像素，计算该元素在bottom中的位置
            Slice<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            				nthreads, bottom_data, kForward, num_slices_, slice_size_,
            				bottom_slice_axis, top_slice_axis, offset_slice_axis, top_data);
            offset_slice_axis += top_slice_axis;
        }
    }
    
    template <typename Dtype>
    void SliceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0] || top..size() == 1) {return;}
        int offset_slice_axis = 0;
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int bottom_slice_axis = bottom[0]->shape(slice_axis_);
        
        const bool fForward = false;
        for(int i = 0; i < top.size(); ++i){
            const Dtype* top_diff = top[i]->gpu_diff();
            const int top_slice_axis = top[i]->shape(slice_axis_);
            const int top_slice_size = top_slice_axis * slice_size_;
            const int nthreads = top_slice_size * num_slices_;
            
            Slice<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            			nthreads, top_diff, kForward, num_slices_, slice_size_,
            			bottom_slice_axis, top_slice_axis, offset_slice_axis, 		
                		bottom_diff);
            offset_slice_axis += top_slice_axis;
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(SliceLayer);
}
```

