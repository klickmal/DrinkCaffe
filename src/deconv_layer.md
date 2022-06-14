# Deconvolution Layer
Deconvolution layer可以看做是Convolution layer的逆过程，

Deconvolution layer在源文件deconv_layer.hpp, deconv_layer.cpp和deconv_layer.cu中实现
**deconv_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe{
	template <typename Dtype>
    class DeconvolutionLayer: public BaseConvolutionLayer<Dtype>{
    public:
    	explicit DeconvolutionLayer(const LayerParameter& param)
          : BaseConvolutionLayer<Dtype>(param){}
      	virtual inline const char* type() const {return "Deconvolution";}
    
   	protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& top);
        
        virtual inline bool reverse_dimensions() {return true;}
        virtual void compute_output_shape();
    };
}
```

**deconv_layer.cpp**

```c++
#include <vector>
#include "caffe/layers/deconv_layer.hpp"

namespace caffe{
	template <typename Dtype>
    void DeconvolutionLayer<Dtype>::compute_out_shape(){
        const int* kernel_shape_data = this->kernel_shape_.cpu_data();
        const int* stride_data = this->stride_.cpu_data();
        const int* pad_data = this->pad_.cpu_data();
        const int* dilation_data = this->dilation_.cpu_data();
        
        this->output_Sahpe_.clear();
        int shape_offset_size = this->
            layer_param_.convolution_param().shape_offset_size();
        
        if(shape_offset_size > 0){
            CHECK_EQ(shape_offset_size, this->num_spatial_axes_);
        }
        
        // 计算num_spatial_axes_: 对于[n,c,h,w],那么num_spatial_axes+=2
        // const int first_spatial_axis = channel_axis_ + 1;
  		// const int num_axes = bottom[0]->num_axes();
  		// num_spatial_axes_ = num_axes - first_spatial_axis;
        for(int i=0; i < this->num_spatial_axes_; ++i){
            // i + 1 to skip channel axis
            const int input_dim = this->input_shape(i + 1);
            const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
            
            const int output_dim = shape_offset_size==0 ? stride_data[i]*(input_dim-1)
                + kernel_extent - 2*pad_data[i] : stride_data[i] * (input_dim - 1)
                + kernel_extent - 2*pad_data[i] + this->layer_param_.convolution_param().
                shape_offset(i);
            
            this->output_shape_.push_back(output_dim);
        }    
    }
    
    template <typename Dtype>
    void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top){
        const Dtype* weight = this->blobs_[0]->cpu_data();
        for(int i=0; i < bottom.size(); ++i){
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* top_data = top[i]->mutable_cpu_data();
            for(int n=0; n < this->num_; ++n){
                this->backward_cpu_gemm(bottom_data + n*this->bottom_dim_, weight, 
                                       top_data + n * this->top_dim_);
                if(this->bias_term_){
                    const Dtype* bias = this->blobs_[1]->cpu_data();
                    this->forward_cpu_bias(top_data + n * this->top_dim, bias);
                }
            }
        }
    }
    
    template <typename Dtype>
    void DeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom){
		const Dtype* weight = this->blobs_[0].cpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
        for(int i=0; i<top.size(); ++i){
            const Dtype* top_diff = top[i]->cpu_diff();
            const Dtype* bottom_data = bottom[i]->cpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
            
            if(this->bias_term_ && this->param_propagate_dowm[1]){
                Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
                for(int n=0; n < this->num_; ++n){
                    this->backward_cpu_bias(bias_diff, top_diff + n*this->top_dim_);
                }
            }
            
            if(this->param_propagate_down_[0] || propagate_down[i]){
                for(int n=0; n < this->num_; ++n){
                    if(this->param_propagate_down_[0]){
                        this->weight_cpu_gemm(top_diff + n*this->top_dim_,
                                             bottom_data + n*this->bottom_dim_,
                                             weight_diff);
                    }
                    if(propagate_down[i]){
                        this->forward_cpu_gemm(top_diff + n*this->top_dim_,
                                              weight, bottom_diff + n*this->bottom_dim_,
                                              this->param_propagate_down[0]);
                    }
                }
            }
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(DeconvolutionLayer);
#endif
    
    INSTANTIATE_CLASS(DeconvolutionLayer);
    REGISTER_LAYER_CLASS(Deconvolution);
    
}

```

**deconv_layer.cu**

```c++
#include <vector>

#include "caffe/layers/deconv_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    void DeconvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top){
        const Dtype* weight = this->blobs_[0]->gpu_data();
        for(int i=0; i<bottom.size(); ++i){
            const Dtype* bottom_data = bottom[i]->gpu_data();
            Dtype* top_data = top[i]->mutable_gpu_data();
            for(int n=0; n<this->num_; ++n){
                this->backward_gpu_gemm(bottom_data + n*this->bottom_dim_, weight,
                                       top_data + n*this->top_dim_);
                if(this->bias_term_){
                    const Dtype* bias = this->blobs_[1]->gpu_data();
                    this->forward_gpu_bias(top_data + n*this->top_dim_, bias);
                }
            }
        }
    }
    
    template <typename Dtype>
    void DeconvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                                const vector<Blob<Dtype>*>& bottom){
        const Dtype* weight = this->blobs_[0]->gpu_data();
        Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
        for(int i = 0; i < top.size(); ++i){
            const Dtype* top_diff = top[i]->gpu_diff();
            const Dtype* bottom_data = bottom[i]->gpu_data();
            Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
            // Bias gradient, if necessary.
            if(this->bias_term_ && this->param_propagate_down_[1]){
                Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
                for(int n=0; n < this->num_; ++n){
                    this->backward_gpu_bias(bias_diff, top_diff + n*this->top_dim_);
                }
            }
            
            if(this->param_propagate_down_[0] || propagate_down[i]){
                for(int n=0; n < this->num_; ++n){
                    // Gradient w.r.t. weight. Note that we will accumulate diffs.
                    if(this->param_prapagate_down_[0]){
                        this->weight_gpu_gemm(top_diff + n*this->top_dim_,
                                             bottom_data + n*this->bottom_dim_, weight_diff);
                    }
                    
                    // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        			// we might have just computed above.
                    if(propagate_down[i]){
                        this->forward_gpu_gemm(top_diff + n*this->top_dim_, weight,
                                               bottom_diff + n*this->bottom_dim_,
                                               this->param_propagate_down_[0]);
                    }
                }
            }
        }
    }
    
	INSTANTIATE_LAYER_GPU_FUNCS(DeconvolutionLayer);
}
```

