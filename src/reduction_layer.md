# Reduction Layer
Reduction Layer: 

* Compute "reductions" -- operations that return a scalar output Blob for an input Blob of arbitrary size, such as the sum, absolute sum, and sum of squares.

reduction layer在reduction_layer.hpp, reduction_layer.cpp和reduction.cu中实现

**reduction_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class ReductionLayer: public Layer<Dtype>{
    public:
        explicit ReductionLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "Reduction";}
        virtual inline int ExactNumBottomBlobs() const {return 1;}
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
        // the reduction operation performed by the layer
        ReductionParameter_ReductionOp op_;
        // a scalar coefficient applied to all outputs
        Dtype coeff_;
        // the index of the first input axis to reduce
        int axis_;
        // the number of reductions performed
        int num_;
        int dim_;
        
        Blob<Dtype> sum_multiplier_;
    };
}
```

**reduction_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
	template <typename Dtype>
    void ReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top){
        op_ = this->layer_param_.reduction_param().operation();
    }
    
    template <typename Dtype>
    void ReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
        axis_ = bottom[0]->CanonicalAxisIndex(
            		this->layer_param_.reduction_param().axis());
        vector<int> top_shape(bottom[0]->shape().begin(),
                             bottom[0]->shape().begin() + axis_);
        top[0]->Reshape(top_shape);
        num_ = bottom[0]->count(0, axis_);
        dim_ = bottom[0]->count(axis_);
        CHECK_EQ(num_, top[0]->count());
        
        if(op_ == ReductionParameter_ReductionOps_SUM ||
           op_ == ReductionParameter_ReductionOp_MEAN){
            vector<int> sum_mult_shape(1, dim_);
            sum_multiplier_.Reshape(sum_mult_shape);
            caffe_set(dim_, Dtype(1), sum_multiplier_.mutable_cpu_data());
        }
        coeff_ = this->layer_param().reduction_param().coeff();
    }
    
    template <typename Dtype>
    void ReductionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* mult_data = NULL;
        if(sum_multiplier_.count() > 0){
            mult_data = sum_multiplier_.cpu_data();
        }
        Dtype* top_data = top[0]->mutable_cpu_data();
        for(int i = 0; i < num_; ++i){
            switch (op_){
                case ReductionParameter_ReductionOp_SUM:
                case ReductionParameter_ReductionOp_MEAN:
                    *top_data = caffe_cpu_dot(dim_, mult_data, bottom_data);
                    break;
                case ReductionParameter_ReductionOp_ASUM:
                    *top_data = caffe_cpu_asum(dim_, bottom_data);
                    break;
                case ReductionParameter_RedcutionOp_SUMSQ:
                    *top_data = caffe_cpu_dot(dim_, bottom_data, bottom_data);
                   	break;
                default:
                    LOG(FATAL) << "Unknown reduction op: "
                        << ReductionParameter_ReductionOp_Name(op_);
            }
            bottom_data += dim_;
            ++top_data;
        }
        
        if(op_ == ReductionParameter_RedcutionOp_MEAN){
        	top_data= top[0]->mutable_cpu_data();
            caffe_scal(num_, Dtype(1) / dim_, top_data);
        }
        
        if(coeff_ != Dtype(1)){
            top_data = top[0]->mutable_cpu_data();
            caffe_scal(num_, coeff_, top_data);
        }
    }
    
    template <typename Dtype>
    void ReductionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0]) {return;}
        const Dtype* bottom_data = NULL;
        switch(op_){
            case ReductionParameter_ReductionOp_SUM:
            case ReductionParameter_ReductionOp_MEAN:
                break;
            case ReductionParameter_ReductionOp_ASUM:
            case ReductionParameter_ReductionOp_SUMSQ:
                bottom_data = bottom[0]->cpu_data();
                break;
            default:
                LOG(FATAL) << "Unknown reduction op: "
                    << RedcutionParameter_ReductionOp_Name(op_);
        }
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        
        for(int i = 0; i < num; ++i){
            Dtype bottom_coeff = (*top_diff) * coeff_;
            if(op_ == ReductionParamter_ReductionOp_MEAN){
                bottom_coeff /= dim_;
            }
            
            switch(op_){
                case ReductionParameter_ReductionOp_SUM:
                case RedcutionParameter_ReductionOp_MEAN:
                	caffe_set(dim_, bottom_coeff, bottom_diff);
                case ReductionParameter_ReductionOp_ASUM:
                    caffe_cpu_sign(dim_, bottom_data, bottom_diff);
                    caffe_scal(dim_, bottom_coeff, bottom_diff);
                    break;
                case ReductionParameter_ReductionOp_SUMSQ:
                    caffe_cpu_scale(dim_, 2 * bottom_coeff, bottom_data, bottom_diff);
                    break;
                default:
                    LOG(FATAL)  << "Unknown reduction op: "
                        << ReductionParameter_ReductionOp_Name(op_);
            }
            bottom_data += dim_;
            bottom_diff += dim_;
            ++top_diff;
        }
    }
    
#ifdef CPU_ONLY
    STUB_GPU(ReductionLayer);
#endif
    
    INSTANTIATE_CLASS(ReducitonLayer);
    REGISTER_LAYER_CLASS(Reduction);
}

```

 1. caffe_cpu_dot在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
        const float* y, const int incy) {
      return cblas_sdot(n, x, incx, y, incy);
    }
    
    template <>
    double caffe_cpu_strided_dot<double>(const int n, const double* x,
        const int incx, const double* y, const int incy) {
      return cblas_ddot(n, x, incx, y, incy);
    }
    
    template <typename Dtype>
    Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
      return caffe_cpu_strided_dot(n, x, 1, y, 1);
    }
    
    template
    float caffe_cpu_dot<float>(const int n, const float* x, const float* y);
    
    template
    double caffe_cpu_dot<double>(const int n, const double* x, const double* y);
    ```

    cblas_sdot和cblas_ddot计算两个向量的点积：

    ```c++
    float  cblas_sdot(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST float  *y, OPENBLAS_CONST blasint incy);
    
    double cblas_ddot(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST double *y, OPENBLAS_CONST blasint incy);
    ```

 2. caffe_cpu_asum在math_functions.hpp中声明如下：

    ```c++
    // Returns the sum of the absolute values of the elements of vector x
    template <typename Dtype>
    Dtype caffe_cpu_asum(const int n, const Dtype* x);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    float caffe_cpu_asum<float>(const int n, const float* x) {
      return cblas_sasum(n, x, 1);
    }
    
    template <>
    double caffe_cpu_asum<double>(const int n, const double* x) {
      return cblas_dasum(n, x, 1);
    }
    ```

 3. caffe_scal在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_scal(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_scal<float>(const int N, const float alpha, float *X) {
      cblas_sscal(N, alpha, X, 1);
    }
    
    template <>
    void caffe_scal<double>(const int N, const double alpha, double *X) {
      cblas_dscal(N, alpha, X, 1);
    }
    ```

    ```c++
    void cblas_sscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST float alpha, float *X, 
                     OPENBLAS_CONST blasint incX);
    
    void cblas_dscal(OPENBLAS_CONST blasint N, OPENBLAS_CONST double alpha, double *X, 
                     OPENBLAS_CONST blasint incX);
    ```

 3. caffe_set在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_set(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <typename Dtype>
    void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
      if (alpha == 0) {
        memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
        return;
      }
      for (int i = 0; i < N; ++i) {
        Y[i] = alpha;
      }
    }
    ```

 4. caffe_cpu_sign是定义在math_functions.hpp中的宏函数：

    ```c++
    template<typename Dtype>
    inline int8_t caffe_sign(Dtype val) {
      return (Dtype(0) < val) - (val < Dtype(0));
    }
    
    #define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
      template<typename Dtype> \
      void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
        CHECK_GT(n, 0); CHECK(x); CHECK(y); \
        for (int i = 0; i < n; ++i) { \
          operation; \
        } \
      }
    
    // output is 1 for the positives, 0 for zero, and -1 for the negatives
    DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));
    ```

 5. caffe_cpu_scale在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                                float* y) {
      cblas_scopy(n, x, 1, y, 1);
      cblas_sscal(n, alpha, y, 1);
    }
    
    template <>
    void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                                 double* y) {
      cblas_dcopy(n, x, 1, y, 1);
      cblas_dscal(n, alpha, y, 1);
    }
    ```

**reduction_layer.cu**

```C++
#include <vector>

#include "caffe/layers/reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

#define sign(x) (Dtype(0) < (x)) - ((x) < Dtype(0))

	template <typename Dtype>
	__global__ void SUMForward(const int num, const int dim, 
							const Dtype* bottom_data, Dtype* top_data){
		CUDA_KERNEL_LOOP(index, num){
			top_data[index] = Dtype(0);
			for(int d = 0; d < dim; d++){
				top_data[index] += bottom_data[index * dim + d];
			}
		}
	}
	
	template <typename Dtype>
    __global__ void MeanForward(const int num, const int dim,
                               const Dtype* bottom_data, Dtype* top_data){
        CUDA_KERNEL_LOOP(index, num){
            top_data[index] = Dtype(0);
            for(int d = 0; d < dim; d++){
                top_data[index] += bottom_data[index * dim + d];
            }
            top_data[index] /= dim;
        }
    }
    
    template <typename Dtype>
    __global__ SUMSQForward(const int num, const int dim, 
                            const Dtype* bottom_data, Dtype* top_data){
        CUDA_KERNEL_LOOP(index, num){
            top_data[index] = Dtype(0);
            for(int d = 0; d < dim; d++){
                top_data[index] += bottom_data[index * dim + d] * 
                    bottom_data[index * dim + d];
            }
        }
    }
    
    template <typename Dtype>
    __global__ void ASUMForward(const int num, const int dim, 
                                const Dtype* bottom_data, Dtype* top_data){
        CUDA_KERNEL_LOOP(index, num){
            top_data[index] = Dtype(0);
            for(int d = 0; d < dim; d++){
                top_data[index] += sign(bottom_data[index * dim + d]);
            }
        }
    }
    
    template <typename Dtype>
    void ReductionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        
        switch(op_){
            case ReductionParameter_ReductionOp_SUM:
                SUMForward<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(
                				num_, dim_, bottom_data, top_data);
            case ReductionParameter_ReductionOp_MEAN:
                MeanForward<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(
                				num_, dim_, bottom_data, top_data);
                break;
            case RedcutionParameter_ReductionOp_ASUM:
                ASUM<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(
                    			num_, dim_, bottom_data, top_data);
                break;
            case ReductionParameter_ReductionOp_SUMSQ:
                SUMSQ<Dtype><<<CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS>>>(
                				num_, dim_, bottom_data, top_data);
            	break;
            default:
                LOG(FATAL) << "Unknown reduction op: " 
                    << ReductionParameter_ReductionOp_Name(op_);   
        }
        if(coeff_ != Dtype(1)){
            top_data = top[0]->mutbale_gpu_data();
            caffe_gpu_scal(num_, coeff_, top_data);
        }
    }
    
    template <typename Dtype>
    __global__ void SUMBackward(const int num, const int dim, 
                               const Dtype* top_diff, Dtype* bottom_diff){
        CUDA_KERNEL_LOOP(index, num * dim){
            int n = index / dim;
            bottom_diff[index] = top_diff[n];
        }
    }
    
    template <typename Dtype>
    __global__ void MeanBackward(const int num, const int dim, 
                                 const Dtype* top_diff, Dtype* bottom_diff){
        CUDA_KERNEL_LOOP(index, num * dim){
            int n = index / dim;
            bottom_diff[index] = top_diff[n] / dim;
        }
    }
    
    template <typename Dtype>
    __global__ void ASUMBackward(const int num, const int dim,
                                const Dtype* top_diff, const Dtype* bottom_data,
                                Dtype* bottom_diff){
        CUDA_KERNEL_LOOP(index, num * dim){
            int n = index / dim;
            bottom_diff[index] = top_diff[n] * sign(bottom_data[index]);
        }
    }
    
    template <typename Dtype>
    __global__ SUMSQBackward(const int num, const int dim, 
                            const Dtype* top_diff, const Dtype* bottom_data){
        CUDA_KERNEL_LOOP(index, num * dim){
            int n = index / dim;
            bottom_diff[index] = top_diff[n] * bottom_data[index] * 2;
        }
    }
    
    template <typename Dtype>
    void ReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom){
        if(!propagate_down[0]) {return;}
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        
        switch(op_){
            case ReductionParameter_ReductionOp_SUM:
                SUMBackward<Dtype><<< CAFFE_GET_BLOCKS(num_*dim_), CAFFE_CUDA_NUM_THREADS>>>
         							(num_, dim_, top_diff, bottom_diff);
            case ReductionParameter_ReductionOp_MEAN:
                MeanBackward<Dtype><<< CAFFE_GET_BLOCKS(num_*dim_), CAFFE_CUDA_NUM_THREADS>>>
                    				(num_, dim_, top_diff, bottom_diff);
                break;
            case RedcutionParameter_ReductionOp_ASUM:
                ASUMBackward<Dtype><<< CAFFE_GET_BLOCKS(num_*dim_), CAFFE_CUDA_NUM_THREADS>>>
                    				(num_, dim_, top_diff, bottom_data, bottom_diff);
                break;
            case ReductionParameter_ReductionOp_SUMSQ:
                SUMSQBackward<Dtype><<< CAFFE_GET_BLOCKS(num_*dim_), CAFFE_CUDA_NUM_THREADS>>>
                    				(num_, dim_, top_diff, bottom_data, bottom_diff);
                break;
            default:
                LOG(FATAL) << "Unknown reduction op: "
                    << ReductionParameter_ReductionOp_Name(op_);
        }
        if(coeff_ != Dtype(1)){
            caffe_gpu_scal(num_*dim_, coeff_, bottom_diff);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUNCS(ReductionLayer);
}
```

 1. caffe_gpu_scal在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <>
    void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
    	CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    }
    
    template <>
    void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
    	CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
    }
    ```

    
