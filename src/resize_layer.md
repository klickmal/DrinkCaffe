# Resize Layer
Resize Layer利用双线性插值resize输入blobs。

resize layer在resize_layer.hpp, resize_layer.cpp和resize_layer.cu中实现

**resize_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    class ResizeLayer: public Layer<Dtype>{
    public:
        explicit ResizeLayer(const LayerParameter& param):
        	: Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Resiz";}
        virtual inline int ExactNumTopBlobs() const {return 1;}
        virtual inline int ExactNumBottomBlobs() const {return 1;}
    
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
        
        vector<Blob<Dtype>*> locs_;
        int out_height_;
        int out_width_;
        int out_channels_;
        int out_num_;
    };
}
```

**resize_layer.cpp**

```c++
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/util_img.hpp"

namespace caffe{
    
	template <typename Dtype>
    void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        // Configure kernel size, padding, stride and inputs
        ResizeParameter resize_param = this->layer_param_.resize_param();
        
        bool is_pyramid_test = resize_param.is_pyramid_test();
        
        if(is_pyramid_test == false){
            CHECK(resize_param.has_height()) << "output height is required";
            CHECK(resize_param.has_width()) << "output width is required";
            this->out_height_ = resize_param.height();
            this->out_width_ = resize_param.width();
        }
        else{
            CHECK(resize_param.has_height_scale()) << "output height scale is required";
            CHECk(resize_param.has_width_scale()) << "output width scale is required";
            int in_height = bottom[0]->height();
            int in_width = bottom[0]->width();
            this->out_height_ = int(resize_param.out_height_scale()*in_height);
            this->out_width_ = int(resize_param.out_width_scale()*in_width);
        }
        
        for(int i=0; i<4; ++i){
            this->locs_.push_back(new Blob<Dtype>);
        }
    }
    
    template <typename Dtype>
    void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top){
        ResizeParameter resize_param = this->layer_param_.resize_param();
        
        bool is_pyramid_test = resize_param.is_pyramid_test();
        if(is_pyramid_test == false){
            this->out_height = resize_param.height();
            this->out_width = resize_param.width();
        }
        else{
            int in_height = bottom[0]->height();
            int in_width = bottom[0]->widht();
            this->out_height_ = int(resize_param.out_height_scale()*in_height);
            this->out_width_ = int(resize_param.out_width_scale()*in_width);
        }
        
        // 只在height和width维度上进行resize操作，在batch和channel维度上保持不变
        this->out_num_ = bottom[0]->num();
        this->out_channels = bottom[0]->channels();
        top[0]->Reshape(out_num_, out_channels_, out_height_, out_width_);
        
        // locs_为什么是4，因为双线性插值是利用四个像素点插值计算出一个像素，所以需要记录
        // 四个角点的index和双线性插值时的权重
        for(int i=0; i<4; ++i){
            this->locs_[i]->Reshape(1, 1, out_height_, out_width_);
        }
    }
    
    template <typename Dtype>
    void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
        ResizeBlob_cpu(bottom[0], top[0]);
    }
    
    template <typename Dtype>
    void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom){
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        Dtype* top_diff = top[0]->mutable_cpu_diff();
        
        const Dtype* loc1 = this->locs_[0]->cpu_data();
        const Dtype* weight1 = this->locs_[0]->cpu_diff();
        const Dtype* loc2 = this->locs_[1]->cpu_data();
        const Dtype* weight2 = this->locs_[1]->cpu_diff();
        const Dtype* loc3 = this->locs_[2]->cpu_data();
        const Dtype* weight3 = this->locs_[3]->cpu_diff();
        const Dtype* loc4 = this->locs_[3]->cpu_data();
        const Dtype* weight2 = this->locs_[3]->cpu_diff();
        
        caffe::caffe_st(bottom[0]->count(), Dtype(0), bottom_diff);
        caffe::GetBiLinearResizeMatRules_cpu(bottom[0]->height(), bottom[0]->width(),
                top[0]->height(), top[0]->width(),
                this->locs_[0]->mutable_cpu_data(), this->locs_[0]->mutable_cpu_diff(),
                this->locs_[1]->mutable_cpu_data(), this->locs_[1]->mutable_cpu_diff(),
                this->locs_[2]->mutable_cpu_data(), this->locs_[2]->mutable_cpu_diff(),
                this->locs_[3]->mutable_cpu_data(), this->locs_[3]->mutable_cpu_diff());
        
        for(int n=0; n<this->out_num_; ++n){
            for(int c=0; c<this->out_channels_; ++c){
                int bottom_diff_offset = bottom[0]->offset(n, c);
                int top_diff_offset = top[0]->offset(n, c);
                
                for(int idx = 0; idx < this->out_height_ * this->out_width_; ++idx){
                    bottom_diff[bottom_diff_offset + static_cast<int>(loc1[idx])] +=
                        top_diff[top_diff_offset + idx] * weight1[idx];
                    bottom_diff[bottom_diff_offset + static_cast<int>(loc2[idx])] +=
                        top_diff[top_diff_offset + idx] * weight2[idx];
                    bottom_diff[bottom_diff_offset + static_cast<int>(loc3[idx])] +=
                        top_diff[top_diff_offset + idx] * weight3[idx];
                    bottom_diff[bottom_diff_offset + static_cast<int>(loc4[idx])] +=
                        top_diff[top_diff_offset + idx] * weight4[idx];
                }
            }
        }
    }
 
#ifdef CPU_ONLY
	STUB_GPU(ResizeLayer);
#endif
    
    INSTANTIATE_CLASS(ResizeLayer);
    REGISTER_LAYER_CLASS(Resize);
}

```

 1. ResizeBlob_cpu在util_img.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
    		Blob<Dtype>* dst, const int dst_n, const int dst_c);
    
    template <typename Dtype>
    void ResizeBlob_cpu(const Blob<Dtype>* src, Blob<Dtype>* dst);
    
    template <typename Dtype>
    void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst,
    		Blob<Dtype>* loc1, Blob<Dtype>* loc2, Blob<Dtype>* loc3, Blob<Dtype>* loc4);
    ```

    ResizeBlob_cpu在util_img.cpp中进行实例化

    ```c++
    template <typename Dtype>
    void ResizeBlob_cpu(const Blob<Dtype>* src, const int src_n, const int src_c,
    		Blob<Dtype>* dst, const int dst_n, const int dst_c) {
    
    
    	const int src_channels = src->channels();
    	const int src_height = src->height();
    	const int src_width = src->width();
    	const int src_offset = (src_n * src_channels + src_c) * src_height * src_width;
    
    	const int dst_channels = dst->channels();
    	const int dst_height = dst->height();
    	const int dst_width = dst->width();
    	const int dst_offset = (dst_n * dst_channels + dst_c) * dst_height * dst_width;
    
    
    	const Dtype* src_data = &(src->cpu_data()[src_offset]);
    	Dtype* dst_data = &(dst->mutable_cpu_data()[dst_offset]);
    	BiLinearResizeMat_cpu(src_data,  src_height,  src_width,
    			dst_data,  dst_height,  dst_width);
    }
    
    template void ResizeBlob_cpu(const Blob<float>* src, const int src_n, const int src_c,
    		Blob<float>* dst, const int dst_n, const int dst_c);
    template void ResizeBlob_cpu(const Blob<double>* src, const int src_n, const int src_c,
    		Blob<double>* dst, const int dst_n, const int dst_c);
    
    
    template <typename Dtype>
    void ResizeBlob_cpu(const Blob<Dtype>* src,Blob<Dtype>* dst)
    {
    	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
    	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";
    
    	for(int n=0;n< src->num();++n)
    	{
    		for(int c=0; c < src->channels() ; ++c)
    		{
    			ResizeBlob_cpu(src,n,c,dst,n,c);
    		}
    	}
    }
    template void ResizeBlob_cpu(const Blob<float>* src,Blob<float>* dst);
    template void ResizeBlob_cpu(const Blob<double>* src,Blob<double>* dst);
    ```

 2. BiLinearResizeMat_cpu在util_img.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void BiLinearResizeMat_cpu(const Dtype* src, const int src_h, const int src_w,
    		Dtype* dst, const int dst_h, const int dst_w);
    ```

    在util_img.cpp中进行实例化

    ```c++
    template <typename Dtype>
    void BiLinearResizeMat_cpu(const Dtype* src, const int src_height, const int src_width,
    		Dtype* dst, const int dst_height, const int dst_width)
    {
    	const Dtype scale_w = src_width / (Dtype)dst_width;
    	const Dtype scale_h = src_height / (Dtype)dst_height;
    	Dtype* dst_data = dst;
    	const Dtype* src_data = src;
    
    	for(int dst_h = 0; dst_h < dst_height; ++dst_h){
    		Dtype fh = dst_h * scale_h;
    
    		int src_h = std::floor(fh);
    
    		fh -= src_h;
    		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
    		const Dtype w_h1 = std::abs(fh);
    
    		const int dst_offset_1 =  dst_h * dst_width;
    		const int src_offset_1 =  src_h * src_width;
    
    		Dtype* dst_data_ptr = dst_data + dst_offset_1;
    		
            // 计算双线性插值结果 f(x,y)=f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy
    		for(int dst_w = 0 ; dst_w < dst_width; ++dst_w){
    
    			Dtype fw = dst_w * scale_w;
    			int src_w = std::floor(fw);
    			fw -= src_w;
    			const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
    			const Dtype w_w1 = std::abs(fw);
    
    
    			Dtype dst_value = 0;
    
    			const int src_idx = src_offset_1 + src_w;
    			dst_value += (w_h0 * w_w0 * src_data[src_idx]);
    			int flag = 0;
                
                // 处理右边界
    			if (src_w + 1 < src_width){
    				dst_value += (w_h0 * w_w1 * src_data[src_idx + 1]);
    				++flag;
    			}
                
                // 处理下边界
    			if (src_h + 1 < src_height){
    				dst_value += (w_h1 * w_w0 * src_data[src_idx + src_width]);
    				++flag;
    			}
    			
                //处理右下边界
    			if (flag>1){
    				dst_value += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
    			// ++flag;
    			}
    			*(dst_data_ptr++) = dst_value;
    		}
    	}
    
    }
    
    
    template void BiLinearResizeMat_cpu(const float* src, const int src_height, const int src_width,
    		float* dst, const int dst_height, const int dst_width);
    
    template void BiLinearResizeMat_cpu(const double* src, const int src_height, const int src_width,
    		double* dst, const int dst_height, const int dst_width);
    ```

    **此处需要看双线性插值的基本原理。**

 3. GetBiLinearResizeMatRules_cpu在util_img.hpp声明如下：

    ```c++
    template <typename Dtype>
    void GetBiLinearResizeMatRules_cpu(  const int src_h, const int src_w,
    		  const int dst_h, const int dst_w,
    		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
    		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);
    ```

    在util_img.cpp中进行实例化

    ```c++
    template <typename Dtype>
    void GetBiLinearResizeMatRules_cpu( const int src_height, const int src_width,
    		 const int dst_height, const int dst_width,
    		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
    		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
    {
    	const Dtype scale_w = src_width / (Dtype)dst_width;
    	const Dtype scale_h = src_height / (Dtype)dst_height;
    
    
    	int loop_n = dst_height * dst_width;
    
    
    	for(int i=0 ; i< loop_n; i++)
    	{
    		int dst_h = i /dst_width;
    		Dtype fh = dst_h * scale_h;
    		int src_h ;
    		if(typeid(Dtype).name() == typeid(double).name())
    			 src_h = floor(fh);
    		else
    			 src_h = floorf(fh);
    
    		fh -= src_h;
    		const Dtype w_h0 = std::abs((Dtype)1.0 - fh);
    		const Dtype w_h1 = std::abs(fh);
    
    		const int dst_offset_1 =  dst_h * dst_width;
    		const int src_offset_1 =  src_h * src_width;
    
    		int dst_w = i %dst_width;
    		Dtype fw = dst_w * scale_w;
    
    		int src_w ;
    		if(typeid(Dtype).name() == typeid(double).name())
    			src_w = floor(fw);
    		else
    			src_w = floorf(fw);
    
    		fw -= src_w;
    		const Dtype w_w0 = std::abs((Dtype)1.0 - fw);
    		const Dtype w_w1 = std::abs(fw);
    
    		const int dst_idx = dst_offset_1 + dst_w;
    //		dst_data[dst_idx] = 0;
    
    		const int src_idx = src_offset_1 + src_w;
    
    		loc1[dst_idx] = static_cast<Dtype>(src_idx);
    		weight1[dst_idx] = w_h0 * w_w0;
    
    
    		loc2[dst_idx] = 0;
    		weight2[dst_idx] = 0;
    
    		weight3[dst_idx] = 0;
    		loc3[dst_idx] = 0;
    
    		loc4[dst_idx] = 0;
    		weight4[dst_idx] = 0;
    
    		if (src_w + 1 < src_width)
    		{
    			loc2[dst_idx] = static_cast<Dtype>(src_idx + 1);
    			weight2[dst_idx] = w_h0 * w_w1;
    //			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
    		}
    
    		if (src_h + 1 < src_height)
    		{
    //			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
    			weight3[dst_idx] = w_h1 * w_w0;
    			loc3[dst_idx] = static_cast<Dtype>(src_idx + src_width);
    		}
    
    		if (src_w + 1 < src_width && src_h + 1 < src_height)
    		{
    			loc4[dst_idx] = static_cast<Dtype>(src_idx + src_width + 1);
    			weight4[dst_idx] = w_h1 * w_w1;
    //			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
    		}
    
    	}
    
    }
    
    template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
    		 const int dst_height, const int dst_width, float* loc1, float* weight1, float* loc2, 
             float* weight2, float* loc3, float* weight3, float* loc4, float* weight4);
    
    template void GetBiLinearResizeMatRules_cpu(  const int src_height, const int src_width,
    		 const int dst_height, const int dst_width, double* loc1, double* weight1, double* loc2, 
             double* weight2, double* loc3, double* weight3, double* loc4, double* weight4);
    
    ```

**resize_layer.cu**

```c++
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"
#include "caffe/util/util_img.hpp"

namespace caffe{
    
   template <typename Dtype>
   void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
       ResizeBlob_gpu(bottom[0], top[0]);
   }
    
   template <typename Dtype>
   __global__ void kernel_ResizeBackward(const int nthreads, const Dtype* top_diff,
                      const int top_step, Dtype* bottom_diff, const int bottom_step, 
                      const Dtype* loc1, const Dtype* weight1, const Dtype* loc2,
                      const Dtype* weight2, const Dtype* loc3, const Dtype* weight3,
                      const Dtype* loc4, const Dtype* weight4){
       CUDA_KERNERL_LOOP(index, n){
           int bottom_diff_offset = bottom_step * index;
           int top_diff_offset = top_step * index;
           
           for(int idx = 0; idx < top_step; ++idx){
               bottom_diff[bottom_diff_offset + int(loc1[idx])] += 		  														top_diff[top_diff_offset + idx] * weight1[idx];
               bottom_diff[bottom_diff_offset + int(loc2[idx])] +=
                   	top_diff[top_diff_offset + idx] * weight2[idx];
               bottom_diff[bottom_diff_offset + int(loc3[idx])] +=
                   	top_diff[top_diff_offset + idx] * weight3[idx];
               bottom_diff[bottom_diff_offset + int(loc4[idx])] += 
                   	top_diff[top_diff_offset + idx] * weight4[idx];
           }
       }
   }
    
    template <typename Dtype>
    void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom){
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        Dtype* top_diff = top[0]->mutable_gpu_diff();
        
        const Dtype* loc1 = this->locs_[0]->gpu_data();
        const Dtype* weight1 = this->locs_[0]->gpu_diff();
        const Dtype* loc2 = this->locs_[1]->gpu_data();
        const Dtype* weight2 = this->locs_[1]->gpu_diff();
        const Dtype* loc3 = this->locs_[2]->gpu_data();
        const Dtype* weight3 = this->locs_[2]->gpu_diff();
        const Dtype* loc4 = this->locs_[3]->gpu_data();
        const Dtype* weight4 = this->locs_[3]->gpu_diff();
        
        caffe::caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
        
        caffe::GetBiLinearResizeMatRules_gpu(bottom[0]->height(), bottom[0]->width(), 
                 top[0]->height(), top[0]->width(),
                 this->locs_[0]->mutable_gpu_data(), this->locs_[0]->mutable_gpu_diff(),
                 this->locs_[1]->mutable_gpu_data(), this->locs_[1]->mutable_gpu_diff(),
                 this->locs_[2]->mutable_gpu_data(), this->locs_[2]->mutable_gpu_diff(),
                 this->locs_[3]->mutable_gpu_data(), this->locs_[3]->mutable_gpu_diff());
        
        const int top_step = top[0]->offset(0,1);
        const int bottom_step = bottom[0]->offset(0,1);
        
        int loop_n = this->out_num_*this->out_channels_;
        
        kernel_ResizeBackward<Dtype><<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS>>>(loop_n, top_diff, 	 	    top_step, bottom_diff, bottom_step, loc1, weight1, loc2, weight2, loc3, weight3, loc4, weight4);
                                                                                           		 		   			CUDA_POST_KERNEL_CHECK;
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);
}
```

 1. ResizeBlob_gpu在util_img.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst, Blob<Dtype>* loc1, Blob<Dtype>* loc2, 		
                        Blob<Dtype>* loc3, Blob<Dtype>* loc4);
    
    template <typename Dtype>
    void ResizeBlob_gpu(const Blob<Dtype>* src, Blob<Dtype>* dst);
    
    template <typename Dtype>
    void ResizeBlob_gpu(const Blob<Dtype>* src, const int src_n, const int src_c, Blob<Dtype>* dst, 
                        const int dst_n, const int dst_c);
    
    template <typename Dtype>
    void BiLinearResizeMat_gpu(const Dtype* src, const int src_h, const int src_w, Dtype* dst, 
                               const int dst_h, const int dst_w);
    ```

    在util_img.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void kernel_ResizeBlob(const int nthreads,const int num,const int channels, const Dtype* src, const int src_height, const int src_width,
    		Dtype* dst, const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w)
    {
    	CUDA_KERNEL_LOOP(index, nthreads) {
    		int i = index %( dst_height * dst_width);
    		int c = (index/(dst_height * dst_width))%channels;
    		int n = (index/(dst_height * dst_width))/channels;
    		int src_offset = (n * channels + c) * src_height * src_width;
    		int dst_offset = (n * channels + c) * dst_height * dst_width;
    
    		const Dtype* src_data = src+src_offset;
    		Dtype* dst_data = dst+dst_offset;
    
    		int dst_h = i /dst_width;
    		Dtype fh = dst_h * scale_h;
    		const int src_h = floor(fh);
    		fh -= src_h;
    		const Dtype w_h0 = std::abs(1.0f - fh);
    		const Dtype w_h1 = std::abs(fh);
    
    		const int dst_offset_1 =  dst_h * dst_width;
    		const int src_offset_1 =  src_h * src_width;
    
    		int dst_w = i %dst_width;
    		Dtype fw = dst_w * scale_w;
    		const int src_w = floor(fw);
    		fw -= src_w;
    		const Dtype w_w0 = std::abs(1.0f - fw);
    		const Dtype w_w1 = std::abs(fw);
    
    		const int dst_idx = dst_offset_1 + dst_w;
    
    
    		const int src_idx = src_offset_1 + src_w;
    		Dtype res = (w_h0 * w_w0 * src_data[src_idx]);
    
    		if (src_w + 1 < src_width)
    			res += (w_h0 * w_w1 * src_data[src_idx + 1]);
    		if (src_h + 1 < src_height)
    			res += (w_h1 * w_w0 * src_data[src_idx + src_width]);
    
    		if (src_w + 1 < src_width && src_h + 1 < src_height)
    			res += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
    
    		dst_data[dst_idx] = res;
    	}
    }
    
    template <typename Dtype>
    void ResizeBlob_gpu(const Blob<Dtype>* src,Blob<Dtype>* dst) {
    
    	CHECK(src->num() == dst->num())<<"src->num() == dst->num()";
    	CHECK(src->channels() == dst->channels())<< "src->channels() == dst->channels()";
    
    	const int src_num = src->num();
    	const int src_channels = src->channels();
    	const int src_height = src->height();
    	const int src_width = src->width();
    
    
    	const int dst_channels = dst->channels();
    	const int dst_height = dst->height();
    	const int dst_width = dst->width();
    
    
    	const Dtype scale_w = src_width / (Dtype)dst_width;
    	const Dtype scale_h = src_height / (Dtype)dst_height;
    	int loop_n = dst_height * dst_width*dst_channels*src_num;
    	const Dtype* src_data = src->gpu_data();
    	Dtype* dst_data = dst->mutable_gpu_data();
    	kernel_ResizeBlob<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>					 				(loop_n,src_num,src_channels,
    				src_data, src_height,src_width,
    				dst_data, dst_height, dst_width,
    				scale_h,scale_w);
    	CUDA_POST_KERNEL_CHECK;
    }
    
    
    
    template void ResizeBlob_gpu(const Blob<float>* src,
    		Blob<float>* dst);
    template void ResizeBlob_gpu(const Blob<double>* src,
    		Blob<double>* dst);
    ```

 2. GetBiLinearResizeMatRules_gpu在util_img.hpp声明如下：

    ```c++
    template <typename Dtype>
    void GetBiLinearResizeMatRules_gpu(  const int src_h, const int src_w, const int dst_h, const int dst_w,
    		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
    		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4);
    ```

    在util_img.cu中进行实例化

    ```c++
    
    template <typename Dtype>
    __global__ void kernel_GetBiLinearResizeMatRules(const int nthreads,  const int src_height, const int src_width,
    		const int dst_height, const int dst_width, const Dtype scale_h, const Dtype scale_w,
    		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
    				Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
    {
    	CUDA_KERNEL_LOOP(index, nthreads)
    	{
    		int dst_h = index /dst_width;
    		Dtype fh = dst_h * scale_h;
    		const int src_h = floor(fh);
    		fh -= src_h;
    		const Dtype w_h0 = std::abs(1.0f - fh);
    		const Dtype w_h1 = std::abs(fh);
    
    		const int dst_offset_1 =  dst_h * dst_width;
    		const int src_offset_1 =  src_h * src_width;
    
    		int dst_w = index %dst_width;
    		Dtype fw = dst_w * scale_w;
    		const int src_w = floor(fw);
    		fw -= src_w;
    		const Dtype w_w0 = std::abs(1.0f - fw);
    		const Dtype w_w1 = std::abs(fw);
    
    		const int dst_idx = dst_offset_1 + dst_w;
    //		dst_data[dst_idx] = 0;
    
    		const int src_idx = src_offset_1 + src_w;
    
    		loc1[dst_idx] = src_idx;
    		weight1[dst_idx] = w_h0 * w_w0;
    
    		loc2[dst_idx] = 0;
    		weight2[dst_idx] = 0;
    
    		weight3[dst_idx] = 0;
    		loc3[dst_idx] = 0;
    
    		loc4[dst_idx] = 0;
    		weight4[dst_idx] = 0;
    
    		if (src_w + 1 < src_width)
    		{
    			loc2[dst_idx] = src_idx + 1;
    			weight2[dst_idx] = w_h0 * w_w1;
    //			dst_data[dst_idx] += (w_h0 * w_w1 * src_data[src_idx + 1]);
    		}
    
    		if (src_h + 1 < src_height)
    		{
    //			dst_data[dst_idx] += (w_h1 * w_w0 * src_data[src_idx + src_width]);
    			weight3[dst_idx] = w_h1 * w_w0;
    			loc3[dst_idx] = src_idx + src_width;
    		}
    
    		if (src_w + 1 < src_width && src_h + 1 < src_height)
    		{
    			loc4[dst_idx] = src_idx + src_width + 1;
    			weight4[dst_idx] = w_h1 * w_w1;
    //			dst_data[dst_idx] += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
    		}
    
    	}
    }
    
    template <typename Dtype>
    void GetBiLinearResizeMatRules_gpu( const int src_height, const int src_width,
    		 const int dst_height, const int dst_width,
    		Dtype* loc1, Dtype* weight1, Dtype* loc2, Dtype* weight2,
    		Dtype* loc3, Dtype* weight3, Dtype* loc4, Dtype* weight4)
    {
    	const Dtype scale_w = src_width / (Dtype)dst_width;
    	const Dtype scale_h = src_height / (Dtype)dst_height;
    
    
    	int loop_n = dst_height * dst_width;
    
    	kernel_GetBiLinearResizeMatRules<Dtype> <<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(
    			loop_n,  src_height,  src_width,
    			dst_height, dst_width, scale_h, scale_w,
    			loc1,  weight1,  loc2,  weight2,
    			loc3,  weight3,   loc4,   weight4);
    	CUDA_POST_KERNEL_CHECK;
    }
    
    template void GetBiLinearResizeMatRules_gpu(  const int src_height, const int src_width,
    		 const int dst_height, const int dst_width,
    		float* loc1, float* weight1, float* loc2, float* weight2,
    				float* loc3, float* weight3, float* loc4, float* weight4);
    
    template void GetBiLinearResizeMatRules_gpu(  const int src_height, const int src_width,
    		 const int dst_height, const int dst_width,
    		double* loc1, double* weight1, double* loc2, double* weight2,
    				double* loc3, double* weight3, double* loc4, double* weight4);
    ```

    
