# Dropout Layer
dropout layer在源文件dropout_layer.hpp, dropout_layer.cpp和dropout_layer.cu中实现
**dropout_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe{
template <typename Dtype>
class DropoutLayer: public NeuronLayer<Dtype>{
    public:
        explicit DropoutLayer(const LayerParameter& param):
            NeuronLayer<Dtype>(param){}

        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const{return "Dropout";}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top);

        Blob<unsigned int> rand_vec_; // 保存伯努利二项分布的随机数变量
        Dtype threshold_; // 数据被dropout的概率
        Dtype scale_; // scale_=1/(1-threshold_);
        unsigned int uint_thres_;
    };
}
```

**dropout_layer.cpp**

```c++
namespace caffe{
    
    template <typename Dtype>
    void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
    {
		NeuronLayer<Dtype>::LayerSetUp(bottom, top);
        threshold_ = this->layer_param_.dropout_param().dropout_ratio();
        DCHECK(threshold_ > 0.);
        DCHECK(threshold_ < 1.);
        
        scale_ = 1. / (1 - threshold_);
        if(this->layer_param_.dropout_param().sqrt_scale()){
            scale_ = sqrt(scale_);
        }
        unit_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
    }
    
    template <typename Dtype>
    void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top)
    {
        NeuronLayer<Dtype>::Reshape(bottom, top);
        
        rand_vec_.Reshape(bottom[0]->shape());
    }
    
    template <typename Dtype>
    void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){
            const Dtype* bottom_data = bottom[0]->cpu_data();
            Dtype* top_data = top[0]->mutable_cpu_data();
        	unsigned int* mask = rand_vec_.mutable_cpu_data();
            const int count = bottom[0]->count();
        
        	if(this->phase_ == TRAIN){
           		caffe_rng_bernoulli(count, 1. - threshold_, mask);
                for(int i = 0; i < count; ++i){
                    top_data[i] = bottom_data[i] * mask[i] *scale_;
                }
            }
        	else{
                caffe_copy(bottom[0]->count(), bottom_data, top_data);
            }
           
        }
	
    template <typename Dtype>
    void DropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& parapagate_down,
                                  const vector<Blob<Dtype>*>& bottom){
            if(propagate_down[0]){
                const Dtype* top_diff = top[0]->cpu_diff();
                Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
                
                if(this->phase_ == TRAIN){
                    const unsigned int* mask = rand_vec_.cpu_data();
                    const int count = bottom[0]->count();
                    
                    for(int i = 0; i < count; ++i){
                        bottom_diff[i] = top_diff[i]*mask[i]*scale_;
                    }
                }
                else{
                    caffe_copy(top[0]->count(), top_diff, bottom_diff);
                }
            }
        }

    #ifdef CPU_ONLY
        STUB_GPU(DropoutLayer);
    #endif

        INSTANTIATE_CLASS(DropoutLayer);
        REGISTER_LAYER_CLASS(Dropout);
}

```

 1. 其中caffe_rng_bernoulli函数在math_functions.hpp中声明：

    ```c++
    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);
    ```

    caffe_rng_bernoulli在math_functions.cpp中实现：

    ```c++
    template <typename Dtype>
    void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
    	CHECK_GE(n, 0);
    	CHECK(r);
    	CHECK_GE(p, 0);
    	CHECK_LE(p, 1);
    	boost::bernoulli_distribution<Dtype> random_distribution(p);
    	boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
    			variate_generator(caffe_rng(), random_distribution);
    	for (int i = 0; i < n; ++i) {
    		r[i] = static_cast<unsigned int>(variate_generator());
    	}
    }
    
    template
    	void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);
    
    template
    	void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);
    ```

    其中caffe_rng在rng.hpp中定义内联函数如下：

    ```
    typedef boost::mt19937 rng_t;
    
    inline rng_t* caffe_rng() {
      return static_cast<caffe::rng_t*>(Caffe::rng_stream().generator());
    }
    ```

 2. caffe_copy在math_functions.hpp声明如下：

    ```c++
    template <typename Dtype>
    void caffe_copy(const int N, const Dtype *X, Dtype *Y);
    ```

    caffe_copy在math_functions.cpp定义实现如下：

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
    
    template void caffe_copy<int>(const int N, const int* X, int* Y);
    template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    		unsigned int* Y);
    template void caffe_copy<float>(const int N, const float* X, float* Y);
    template void caffe_copy<double>(const int N, const double* X, double* Y);
    ```

​	3. **Inverted Dropout**: 当模型使用dropout layer, 训练的时候只有（1-p)的神经元参与训练，预测时，如果所有隐藏层都参与进来，那		么结果相比训练时平均要大1/(1-p)，为避免这种情况，就需要将测试结果乘以(1-p) 使下一层的输入保持不变。而利用inverted 	 		dropout, 就可以在训练的时候将dropout层输出结果放大1/(1-p)，这样就可以使结果的scale保持不变，而在预测的时候就不需要做		额外的操作，更加方便

**dropout_layer.cu**

```c++
#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namesapce caffe{
    template <typename Dtype>
    __global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, 
    const float scale, Dtype* out){
        CUDA_KERNEL_LOOP(index, n){
        out[index] = in[index] * (mask[index] > threshold_) * scale;
		}
    }
    
    template <typename Dtype>
    void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int count = bottom[0]->count();
        
        if(this->phase_ == TRAIN){
            unsigned int* mask = static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
            caffe_gpu_rgn_uniform(count, mask);
            
            DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            	count, bottom_data, mask, uint_thres_, scale_, top_data);
            CUDA_POST_KERNEL_CHECK;
        }
        else{
            caffe_copy(count, bottom_data, top_data);
        }
    }
    
    template <typename Dtype>
    __global__ void DropoutBackward(const int n, const Dtype* in_diff, 
                                  const unsigned int* mask, const unsigned int threshold, 
                                   const float scale, Dtype* out_diff){
        CUDA_KERNEL_LOOP(index, n){
            out_diff[index] = in_diff[index]*scale*(mask[index] > threshold);
        }
    }
    
    template <typename Dtype>
    void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            
            if(this->phase_ == TRAIN){
                const unsigned int* mask = static_cast<const unsigned int*>(rand_vec_.gpu_data());
                const int count = bottom[0]->count();
                
                DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, 											mask, uint_thres_, scale, bottom_diff);
                CUDA_POSE_KERNEL_CHECK;
            }
            else{
                caffe_copy(top[0]->count(), top_diff, bottom_diff);
            }
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);
}
```

