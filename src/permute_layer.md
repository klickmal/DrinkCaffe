# Permute Layer
permute layer在permute_layer.hpp, permute_layer.cpp和permute_layer.cu中实现

**permute_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

   template <typename Dtype>
   void Permute(const int count, Dtype* bottom_data, const bool forward,
               const int* permute_order, const int* old_steps, const int* new_steps,
               const int num_axes, Dtype* top_data);
    
    // permute layer：改变blob数组的order，例如N×C×H×W变换为N×H×W×C，
    // 那么，permute_param order为：0，order：2，order：3，order：1；
    template <typename Dtype>
    class PermuteLayer: public Layer<Dtype>{
    public:
        explicit PermuterLayer(const LayerParameter& param)
            : Layer<Dtype>(param){}
        virtual void LayerSetUp(const Vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vectot<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Permute";}
        virtual inline int ExactBottomBlobs() const {return 1;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        
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
        
        int num_axes_;
        bool need_permute_;
        
        // Use Blob because it is convenient to be accessible in .cu file.
        Blob<int> permute_order_;
        Blob<int> old_steps_;
        Blob<int> new_steps_;
    };
}
```

**permute_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
   	
    // 假设bottom的shape是[N,C,H,W], permute_order交换顺序是[0,2,3,1]
   	// 那么old_steps_为[C*H*W,H*W,W,1], 
    // top_shape为[N,H,W,C]
    // 那么new_steps_为[H*W*C,W*C,C,1]
    template <typename Dtype>
    void Permute(const int count, Dtype* bottom_data, const bool forward,
                const int* permute_order, const int* old_steps, const int* new_steps,
                const int num_axes, Dtype* top_data){
        
        // 第一个for循环就是依次取出置换后各元素在数组中的索引idx
        for(int i=0; i<count; ++i){
            int old_idx = 0;
            int idx = i;
            
            // 第二个for循环计算idx对应的原数据对应的该元素的索引old_idx
            // 仔细琢磨琢磨计算index这个地方
            // 对于原blob的shape是[N,C,H,W], 某一个元素的index为n[i]*C*H*W+c[i]*H*W+h[i]*W+w[i]
            // 那么新blob的shape是[N,H,W,C], 某一个元素的index为n[i]*C*H*W+h[i]*C*W+w[i]*C+c[i]
            // 已知一个元素在新blob中的位置，可以算出[n[i],h[i],w[i],c[i]]
            // 那么可以计算该元素在原blob中的位置
            for(int j=0; j<num_axes; ++j){
                int order = permuter_order[i];
                old_idx += (idx/new_steps[j])*old_steps[order];
                idx %= new_steps[j];
            }
            
            if(forward){
                top_data[i] = bottom_data[old_idx];
            }
            else{
                bottom_data[old_idx] = top_data[i];
            }
        }
    }
    
    template <typename Dtype>
    void PermuteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        PermuteParameter permute_param = this->layer_param_.permute_param();
        CHECK_EQ(bottom.size(), 1);
        
        num_axes_ = bottom[0]->num_axes();
        vector<int> orders;
        
        for(int i=0; i<permute_param.order_size(); ++i){
            int order = permute_param.order(i);
            CHECK_LT(order, num_axes)
                << "order should be less than the input dimension.";
            
            // Iterator to the first element satisfying the condition or last if no such element is found.
            if(std::find(orders.begin(), orders.end(), order) != orders.end()){
                LOG(FATAL) << "there are duplicate orders";
            }
            orders.push_back(order);
        }
        
        // Push the rest orders. And save original step sizes for each axis.
		// 注意所指定的新的索引轴顺序的大小不一定等于num_axes_,例如原来顺序为0,1,2,3;指定前两轴顺序，即交换为1,0,2,3
		// 这时只指定permute_param.order(0)=1,permute_param.order(1)=0即可，也即只需要			 			
        // permute_param.order_size()=2,后两轴无需指定
        for(int i=0; i<num_axes; ++i){
            if(std::find(orders.begin(), orders.end(), i) == orders.end()){
                orders.push_back(i);
            }
        }
        
        CHECK_EQ(num_axes_, orders.size());
        
        need_permute_ = false;
        for(int i=0; i<num_axes_; ++i){
            if(orders[i] != i){
                // 只要有一个轴的顺序改变，则需要置换顺序（即设置need_permute_为true）
                need_permute_ = true;
                break;
            }
        }
        
        vector<int> top_shape(num_axes_, 1);
        permute_order_.Reshape(num_axes_, 1, 1, 1);
        old_steps_.Reshape(num_axes_, 1, 1, 1);
        new_steps_.Reshape(num_axes_, 1, 1, 1);
        
        // 以下三个变量均为blob类，方便.cu文件的实现
        for(int i=0; i<num_axes_; ++i){
            permute_order_.mutable_cpu_data()[i] = orders[i]; // 用于记录顺序后的各轴顺序
            top_shape[i] = bottom[0]->shape(orders[i]);
        }
        top[0]->Reshape(top_shape);
    }
    
    template <typename Dtype>
    void PermuteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
        // 假设bottom的shape是[n,c,h,w], 交换顺序是[0,2,3,1]
        // 那么old_steps_为[c*h*w,h*w,w,1], top_shape为[n,h,w,c]
        vector<int> top_shape;
        for(int i=0; i<num_axes; ++i){
            if(i == num_axes_ - 1){
                old_steps_.mutable_cpu_data()[i] = 1; 
            }
            else{
                old_steps_.mutable_cpu_data()[i] = bottom[0]->count(i+1);
            }
            top_shape.push_back(bottom[0]->shape(permute_order_.cpu_data()[i]));
        }
        
        top[0]->Reshape(top_shape);
        
        // new_steps_为[h*w*c,w*c,c,1]
        for(int i = 0; i < num_axes_; ++i){
            if(i == num_axes_ - 1){
                new_steps_.mutable_cpu_data()[i] = 1;
            }
            else{
                new_steps_.mutable_cpu_data()[i] = top[0]->count(i+1);
            }
        }
    }
    
    template <typename Dtype>
    void PermuteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        if(need_permute_){
            Dtype* bottom_data = bottom[0]->mutable_cpu_data();
            Dtype* top_data = top[0]->mutable_cpu_data();
            
            const int top_count = top[0]->count();
            const int* permute_order = permute_order_.cpu_data();
            const int* old_steps = old_steps.cpu_data();
            const int* new_steps = new_steps.cpu_data();
            
            bool forward = true;
            Permute(top_count, bottom_data, forward, permute_order, old_steps,
                   new_steps, num_axes_, top_data);
        }
        else{
            top[0]->ShareData(*bottom[0]);
        }
    }
    
    template <typename Dtype>
    void PermuteLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        if(need_permute_){
            Dtype* top_diff = top[0]->mutable_cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            
            const int top_count = top[0]->count();
            const int* permute_order = permute_order_.cpu_data();
            const int* old_steps = old_steps_.cpu_data();
            const int* new_steps = new_steps_.cpu_data();
            
            bool forward = false;
            Permute(top_count, bottom_diff, forward, permute_order, old_steps,
                   new_steps, num_axes_, top_diff);
        }
        else{
            bottom[0]->ShareDiff(*top[0]);
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(PermuteLayer);
#endif
    
    INSTANTIATE_CLASS(PermuteLayer);
    REGISTER_LAYER_CLASS(Permute);
}

```

**permute_layer.cu**

```c++
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/permute_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    template <typename Dtype>
    __global__ void PermuteKernel(const int nthreads, Dtype* const bottom_data, 
                                  const bool forward, const int* permute_order,
                                 const int* old_steps, const int* new_steps,
                                 const int num_axes, Dtype* const top_data){
        CUDA_KERNEL_LOOP(index, nthreads){
            int temp_idx = index;
            int old_idx = 0;
            for(int i=0; i<num_axes; ++i){
                int order = permute_order[i];
                old_idx += (temp_idx/new_steps[i]) * old_steps[order];
                temp_idx %= new_steps[i];
            }
            if(forward){
                top_data[index] = bottom_data[old_idx];
            }
            else{
                bottom_data[old_idx] = top_data[index];
            }
        }
    }
    
    template <typename Dtype>
    void PermuteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        if(need_permute_){
            Dtype* bottom_data = bottom[0]->mutable_gpu_data();
            Dtype* top_data = top[0]->mutable_gpu_data();
            
            int count = top[0]->count();
            const int* permute_order = permute_orer_.gpu_data();
            const int* new_steps = new_steps_.gpu_data();
            const int* old_steps = old_steps.gpu_data();
            
            bool forward = true;
            PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            		count, bottom_data, forward, permute_order, old_steps, new_steps,
            		num_axes_, top_data);
            CUDA_POST_KERNEL_CHECK;
        }
       else{
           top[0]->ShareData(*bottom[0]);
       }
    }
    
    template <typename Dtype>
    void PermuteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        if(need_permute_){
            Dtype* top_diff = top[0]->mutable_gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int count = bottom[0]->count();
            const int* permute_order = permute_order_.gpu_data();
            const int* new_steps = new_steps_.gpu_data();
            const int* old_steps = old_steps_.gpu_data();
            
            bool forward = false;
            PermuteKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            			count, bottom_diff, forward, permute_order, old_steps,
            			new_steps, num_axes_, top_diff);
            CUDA_POST_KERNEL_CHECK;
        }
        els{
           bottom[0]->ShareDiff(*top[0]); 
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(PermuteLayer);
}
```

