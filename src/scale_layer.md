# Scale Layer
scale layer在scale_layer.hpp，scale_layer.cpp和scale_layer.cu中声明实现。

**scale_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/bias_layer.hpp"

namespace caffe{

    template <typename Dtype>
    class ScaleLayer: public Layer<Dtype>{
    public:
        explicit ScaleLayer(const LayerParameter& param)
            :Layer<Dtype>(param){}
        
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Scale";}
        
        // 获取scale层的输入块的数目。 当输入为2时，bottom[1]用于存在scale的值;
		// 如果输入为1时，scale的值作为本层的学习参数，它会根据梯度值进行更新的
        virtual inline int MinBottomBlobs() const {return 1;}
        virtual inline int MaxBottomBlobs() const {return 2;}
        virtual inline int ExactNumTopBlobs() const {return 1;}
   	
    protected:
    	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
        virtual vois Backeard_gpu(const vector<Blob<Dtype>*>& top,
                                 const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
        
        // 当参数中选择了bias，会在scale的基础上增加一个bias值，通过bias_layer_实现
        shared_ptr<Layer<Dtype>> bias_layer_;
        // bias_layer_的输入
        vector<Blob<Dtype>*> bias_bottom_vec_; 
        // 表示bias是否需要梯度反向传播
        vector<bool> bias_propagate_down_;
         
        int bias_param_id_; // bias层的偏置值在scale层中的bobs_中的第几个位置上
        
        Blob<Dtype> sum_multiplifier_;
        Blob<Dtype> sum_result_;
        blob<Dtype> temp_;
        int axis_;
        int outer_dim_, scale_dim_, inner_dim_;
    };
}
```

**scale_layer.cpp**

```c++
#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
   	template <typename Dtype>
    void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const ScaleParameter& param = this->layer_param_.scale_param();
        // 如果输入只有一个 bottom blob，那么 scale factor 就一个可学习的参数（标量），
		// 并且存储在 ScaleLayer 层中的this->blobs_中
		// 接下来，我们可以通过判断this->blobs_的向量长度
		// 是否为0，来判断scale factor是否被初始化 
        if(bottom.size() == 1 && this->blobs.size() > 0){
            LOG(INFO) << "Skipping parameter initialization";
        }
        else if (bottom.size() == 1){ // 如果scale factor没有初始化
            // scale is a learned parameter; initialize it
            axis_ = bottom[0]->CanonicalAxisIndex(param.axis()); 
            const int num_axes = param.num_axes();
            CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                << "or -1 to extend to the end of bottom[0]";
            if(num_axes >= 0)
            {
                CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
                    << "scale blob's shape extends past bottom[0]'s shape when applies '"
                    << "starting with bottom[0] axis = " << axis_;
            }
            // 我们将参数的个数设置为 1
            this->blobs_.resize(1);
            // 获取可扩展 shape 的 start axis 的迭代器
            const vector<int>::const_interator& shape_start =
                bottom[0]->shape().begin() + axis_;
            // 获取可扩展 shape 的 end axis 的迭代器
            const vector<int>::const_interator& shape_end = 
                (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
            // 获取可扩展的 shape 维度
            vector<int> scale_shape(shape_start, shape_end);
            // 为可学习参数 scale factor 分配空间
            this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
            
            FillParameter filler_param(param.filler());
            // 如果没有 filler 填充器，那么我们就填充全 1
            if(!param.has_filler())
            {
                filler_param.set_type("constant");
                filler_param.set_value(1);
            }
            // 获取 filler 指针
            shared_ptr<Filler<Dtype>> filler(GetFiller<Dtype>(filler_param));
            // 为可学习参数的 scale factor 填充初始化数据
            filler->Fill(this->blobs_[0].get());
        }
        // ????????????????????????????????????????????
        // ????????????????????????????????????????????
        // 如果有bias项
        if(param.bias_term())
        {
            LayerParamter layer_param(this->param_);
            layer_param.set_type("Bias");
            
            BiasParameter* bias_param = layer_param.mutable_bias_param();
            bias_param->set_axis(param.axis());
            
            if(bottom.size() > 1){
                bias_param->set_num_axes(bottom[1]->num_axes());
            } 
            else{
                bias_param->set_num_axes(param.num_axes());
            }
         	bias_param->mutable_filler()->CopyFrom(param.bias_filler());
            bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
            bias_bottom_vec_.resize(1);
            bias_bottom_vec_[0] = bottom[0];
            
            bias_layer_->SetUp(bias_bottom_vec_, top);
            if(this->blobs_.size() + bottom.size() < 3)
            {
                // case: blobs.size == 1 && bottom.size == 1
				// or blobs.size == 0 && bottom.size == 2
                bias_param_id_ = this->blobs_.size();
                this->blobs_.resize(bias_param_id_ + 1);
                this->blobs_[bias_param_id_] = bias_layer_->blobs()[0]; 
            }
            else{
                bias_param_id_ = this->blobs_.size() - 1;
                bias_layer_->blobs()[0] = this->blobs_[bias_param_id_];
            }
            bias_propagate_down_.resize(this->blobs_.size(), true);
        }
        this->param_propagate_down_.resize(this->blobs_.size(), true);
    }
    
    template <typename Dtype>
    void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, 
                                   const vector<Blob<Dtype*>>& top){
        const ScaleParameter& param = this->layer_param_.scale_param();
        Blob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->blob_[0].get();
        
        // 当num_axes的值为0时，表示scale值是一个标量，相当于所有的输入值都使用一个scale值进行缩放操作。
		// 此时，把axis设置为0了，这样的接下来的计算很高效
        axis_ = (scale->num_axes() == 0) ?
            0 : bottom[0]->CanonicalAxisIndex(param.axis());
        CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
            << "scale blob's shape extends past bottom[0]'s shape when applied "
            << "starting with bottom[0] axis = " << axis_;
        for(int i = 0; i < scale->num_axes(); ++i){
            CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
                << "dimension mismatch between bottom[0]->shape(" << axis_ + i
                << ") and scale->shape(" << i << ")";
        }
        outer_dim_ = bottom[0]->count(0, axis_);
        scale_dim_ = scale->count();
        inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
        
        if(bottom[0] == top[0]){
            temp_.ReshapeLike(*bottom[0]);
        }
        else{
            top[0]->ReshapeLike(*bottom[0]);
        }
        
        // sum_result_向量值用于保存在inner_dim上求和之后的值, 求和之后就是一个大小就等于outer_dim * scale_dim.
        sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
        
        const int sum_mult_size = std::max(outer_dim_, inner_dim_);
        sum_multiplier_.Reshape(vector<int>(1, sum_mult_size());
        
        if(sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)){
            caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
        }
        if(bias_layer_){
            bias_bottom_vec_[0] = top[0];
            bias_layer_->Reshape(bias_bottom_vec_, top);
        }
    }
                                
    template <typename Dtype>
    void ScaleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>* &bottom,
                                       const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        if(bottom[0] == top[0]){
            // 如果我们进行的是 in-place 计算，我们需要在覆盖它之前，保存 bootom data
			// 这个操作只对反向传播有用。如果我们不进行反向传播，那么就可以跳过该操作。
			// 但是目前 Caffe 在进行前向传播的时候，并不知道我们是否需要反向传播
            caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), temp_.mutable_cpu_data());
        }
        
        Dtype* scale_data = ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->mutable_cpu_data();
        
        // 限制scale的值
        if(this->layer_param_.scale_param().has_min_value()){
            for(int d=0; d<scale_dim_; d++){
                scale_data[d] = std::max<Dtype>(scale_data[d], 
                                                this->layer_param_.scale_param().min_value());
            }
        }
        if(this->layer_param_.scale_param().has_max_value()){
			for(int d=0; d<scale_dim_; d++){
                scale_data[d] = std::min<Dtype>(scale_data[d], 
                                                this->layer_param_.scale_param().max_value());
            }
        }
        
        Dtype* top_data = top[0]->mutable_cpu_data();
        for(int n=0; n<outer_dim_;++n){
            for(int d=0; d<scale_dim_; ++d){
                // 获取扩展因子
                const Dtype factor = scale_data[d];
                // 使用对应的扩展因子扩展 bottom_data，并且输出到 top_data
                caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
                bottom_data += inner_dim;
                top_data += inner_dim_;
            }
        }
        if(bias_layer_){
        	bias_layer_ ->Forward(bias_bottom_vec_, top);
        }
    }
    
    template <typename Dtype>
    void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                             const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
        if(bias_layer_ && 
           this->param_propagate_down[this->param_propagate_down_.size() - 1]){
            bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
        }
        
        const bool scale_param = (bottom.size() == 1);
        BLob<Dtype>* scale = scale_param ? this->blob_[0].get() : bottom[1];
        if((!scale_param && propagate_down[1] || 
            (scale_param && this->param_propagate_down_[0])){
            const Dtype* top_diff = top[0]->cpu_diff();
            const bool in_place = (bottom[0] == top[0]);
            const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->cpu_data();
            
            const bool is_eltwise = (bottom[0]->count() == scale->count());
            Dtype* product = (is_eltwise ? scale->mutable_cpu_diff():
                              (in_place ? temp_.mutable_cpu_data():bottom[0]->mutable_cpu_diff()));
            caffe_mul(top[0]->count(), top_diff, bottom_data, product);
            
            if(!is_eltwise){
                Dtype* sum_result = NULL;
                if(inner_dim_ == 1){
                    sum_result = product;
                }
                else if(sum_result_.count() == 1){
                    const Dtype* sum_mult = sum_multiplier_.cpu_data();
                    Dtype* scale_diff = scale->mutable_cpu_diff();
                    if(scale_param){
                        Dtype result = caffe_cpu_dot(inner_dim_, product, sum_mult);
                        *scale_diff += result;
                    }
                    else{
                        *scale_diff = caffe_cpu_dot(inner_dim_, product, sum_mult);
                    }
                }
                else{
                    const Dtype* sum_mult = sum_multiplier_.cpu_data();
                    sum_result = (outer_dim == 1) ?
                        scale->mutable_cpu_diff() : sum_result_.mutable_cpu_data();
                    caffe_cpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_, 
                                   Dtype(1), product, sum_mult, Dtype(0), sum_result);
                }
                
                if(outer_dim_ != 1){
                    const Dtype* sum_mult = sum_multiplier_.cpu_data();
                    Dtype* scale_diff = scale->mutable_cpu_diff();
                    if(scale_dim == 1){
                        if(scale_param){
                            Dtype result = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
                            *scale_diff += result;
                        }
                        else{
                           	*scale_diff = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
                        }
                    }
                    else{
                        caffe_cpu_gemv(CblasTrans, outer_dim_, scale_dim_, 
                                       Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                                       scale_diff);
                    }
                }
            }                               
    	}
  		
        // 求bottom[0]的梯度
        if(propagate_down[0]){
            const Dtype* top_diff = top[0]->cpu_diff();
            const Dtype* scale_data = scale->cpu_data();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            for(int n=0; n<outer_dim_; ++n){
                for(int d=0; d<scale_dim_; ++d){
                    const Dtype factor = scale_data[d];
                    caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);
                    bottom_diff += inner_dim_;
                    top_diff += inner_dim_;
                }
            }
        }
    }
  	
#ifdef CPU_ONLY
	STUB_GPU(ScaleLayer);
#endif
	INSTANTIATE_CLASS(ScaleLayer);
    REGISTER_LAYER_CLASS(Scale);
}                             

```

 1. 参数设置

    ```c++
    message ScaleParameter {
    
      optional int32 axis = 1 [default = 1]; //默认要处理的维度
      optional int32 num_axes = 2 [default = 1];//只有一个维度
      optional FillerParameter filler = 3; //alpha可学习的，alpha初始值设置
      optional bool bias_term = 4 [default = false];//是否需要偏置项
      optional FillerParameter bias_filler = 5; //偏置项初始化
    }
    ```

 2. caffe_copy在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_copy(const int N, const Dtype *X, Dtype *Y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <typename Dtype>
    	void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
    		if (X != Y) {
    			if (Caffe::mode() == Caffe::GPU) {
    #ifndef CPU_ONLY
    				// NOLINT_NEXT_LINE(caffe/alt_fn)
    				CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
    #else
    				NO_GPU;
    #endif
    		}
    			else {
    				memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    			}
    	}
    }
    ```

 3. caffe_cpu_scale在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_cpu_scale<float>(const int n, const float alpha, const float *x, float* y) {
    	cblas_scopy(n, x, 1, y, 1);
    	cblas_sscal(n, alpha, y, 1);
    }
    
    template <>
    void caffe_cpu_scale<double>(const int n, const double alpha, const double *x, double* y) {
    	cblas_dcopy(n, x, 1, y, 1);
    	cblas_dscal(n, alpha, y, 1);
    }
    ```

 4. 

**scale_layer.cu**

```c++
#include <cfloat>
#include <vector>

#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
   	template <typename Dtype>
    __global__ void ScaleForward(const int n, const Dtype* in, const Dtype* scale, 
                                const int scale_dim, const int inner_dim, Dtype* out)
    {
        CUDA_KERNEL_LOOP(index, n){
            const int scale_index = (index / inner_dim) % scale_dim;
            out[index] = in[index] * scale[scale_index];
        }
    }
    
    template <typename Dtype>
    __global__ void ScaleBiasForward(const int n, const Dtype* in, const Dtype* scale, 
                        const Dtype* bias, const int scale_dim, const int inner_dim, Dtype* out){
        CUDA_KERNEL_LOOP(index, n){
            const int scale_index = (index / inner_dim) % scale_dim;
            out[index] = in[index] * scale[scale_index] + bias[scale_index];
        }
    }
    
    template <typename Dtype>
    __global__ void TruncationLowerBounded(const int n, Dtype* in_out, Dtype lower_bound){
        CUDA_KERNEL_LOOP(index, n){
            in_out[index] = max(in_out[index], lower_bound);
        }
    }
    
    template <typename Dtype>
    __global__ void TruncationUpperBounded(const int n, Dtype* in_out, Dtype upper_bound){
        CUDA_KERNEL_LOOP(index, n){
            in_out[index] = min(in_out[index], upper_bound);
        }
    }
    
    template <typename Dtype>
    void ScaleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                                       const vector<Blob<Dtype>*>& top){
        const int count = top[0]->count();
        const Dtype* bottom_data = bottom[0]->gpu_data();
        
        if(bottom[0] == top[0]){
            caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), temp_.mutable_gpu_data());
        }
        Dtype* scale_data = ((bottom.size()>1) ? bottom[1]: this->blobs_[0].get())->mutable_gpu_data();
        
        if(this->layer_param_.scale_param().has_min_value()){
            TruncationLowerBounded<Dtype><<< CAFFE_GET_BLOCKS(scale_dim_), CAFFE_CUDA_NUM_THREADS>>>(
            	scale_dim_, scale_data, this->layer_param_.scale_param().min_values());
            CUDA_POST_KERNEL_CHECK;
        }
        if(this->layer_param_.scale_param().has_max_value()){
            TruncationUpperBounded<Dtype><<<CAFFE_GET_BLOCKS(scale_dim_), CAFFE_CUDA_NUM_THREADS>>>(
            	scale_dim_, scale_data, this->layer_param_.scale_param().max_value());
            CUDA_POST_KERNEL_CHECK;
        }
        
        Dtype* top_data = top[0]->mutable_gpu_data();
        
        if(bias_layer_){
            const Dtype* bias_data = this->blobs_[bias_param_id_]->gpu_data();
            ScaleBiasForward<Dtype><<<CAFFE_GET_BLOCK(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
                bottom_data, scale_data, bias_data, scale_dim, inner_dim, top_data);
        }
        else{
            ScaleForward<Dtype><<<CAFFE_GET_BLOCK(count), CAFFE_CUDA_NUM_THREADS>>>(count, 
                bottom_data, scale_data, scale_dim_, inner_dim_, top_data);
        }
    }
    
    template <typename Dtype>
    void ScaleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
    {
        if(bias_layer_ && this->param_propagate_down_[this->param_propagate_down_.size() - 1]){
            bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
        }
        
        const bool scale_param = (bottom.size()==1);
        Blob<Dtype>* scale = scale_param ? this->blobs_[0].get():bottom[1];
        
        if((!scale_param && propagate_down[1]) ||
           (scale_param && this->param_propagate_down_[0])){
            const Dtype* top_diff = top[0]->gpu_diff();
            const bool in_place = (bottom[0] == top[0]);
            const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->gpu_data();
            
            const bool is_eltwise = (bottom[0]->count() == scale->count());
            Dtype* product = (is_eltwise ? scale->mutable_gpu_diff()) :
            		(in_place ? temp_.mutable_gpu_data() : bottom[0]->mutable_gpu_diff());
            caffe_gpu_mul(top[0]->count(), top_diff, bottom_data, product);
            
            if(!eltwise){
                Dtype* sum_result = NULL;
                if(inner_dim_==1){
                    sum_result = product;
                }else if(sum_result_.count()==1){
                    const Dtype* sum_mult = sum_multiplier_.gpu_data();
                    Dtype* scale_diff = scale->mutable_cpu_diff();
                    if(scale_param){
                        Dtype result;
                        caffe_gpu_dot(inner_dim_, product, sum_mult, &result);
                        *scale_diff += result;
                    }
                    else{
                        caffe_gpu_dot(inner_dim_, product, sum_mult, scale_diff);
                    }
                }
                else{
                  const Dtype* sum_mult = sum_multiplier_.gpu_data();
                  sum_result = (outer_dim_ == 1) ? scale->mutable_gpu_diff():sum_result_.mutable_gpu_data();
                  caffe_gpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_, Dtype(1),
                                product, sum_mult, Dtype(0), sum_result);
                }
                if(outer_dim_!=1){
                    const Dtype* sum_mult = sum_multiplier_.gpu_data();
                    if(scale_dim_ == 1){
                        Dtype* scale_diff = scale->mutable_cpu_diff();
                        if(scale_param){
                            Dtype result;
                            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, &result);
                            *scale_diff += result;
                        }
                        else{
                            caffe_gpu_dot(outer_dim_, sum_mult, sum_result, scale_diff);
                        }
                    }
                    else{
                        Dtype* scale_diff =  scale->mutable_gpu_diff();
                        caffe_gpu_gemv(CblasTrans, outer_dim_, scale_dim_, Dtype(1), 
                                       sum_result, sum_mult, Dtype(scale_param), scale_diff);
                    }
                }
            }
        }
        if(propagate_down[0]){
            const int count = top[0]->count();
            const Dtype* top_diff = top[0]->gpu_diff();
            const Dtype* scale_data = bottom[0]->mutable_gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            ScaleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            	count, top_diff, scale_data, scale_dim, inner_dim_, bottom_diff);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(ScaleLayer);
}
```

