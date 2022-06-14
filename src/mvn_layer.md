# MVN Layer
MVN Layer功能：Normalizes the input to have 0-mean and/or unit (1) variance.

Mean-Variance Normalization layer在mvn_layer.hpp, mvn_layer.cpp和mvn_layer.cu中实现

**mvn_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class MVNLayer: public Layer<Dtype>{
    public:
        explicit MVNLayer(const LayerParameter& param)
            : Layer<Dtype>(param) {}
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
        
        virtual inline const char* type() const {return "MVN";}
        virtual inline int ExactNumBottomBlobs() const  {return 1;}
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
        
        Blob<Dtype> mean_, variance_, temp_;
        // sum_multiplier is used to carry out sum using BLAS
        Blob<Dtype> sum_multiplier_;
        Dtype eps_;
    };
}
```

**mvn_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
	template <typename Dtype>
    void MVNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top){
        top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
                       bottom[0]->height(), bottom[0]->width());
        mean_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
        variance_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
        temp_.Reshape(bottom[0]->num(), bottom[0]->channels(), 
                     bottom[0]->height(), bottom[0]->width());
        if(this->layer_param_.mvn_param().across_channels()){
            sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
        }
        else{
            sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
        }
        Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
        caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
        eps_ = this->layer_param_.mvn_param().eps();
    }
    
    template <typename Dtype>
    void MVNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        int num;
        if(this->layer_param_.mvn_param().across_channels())
            num = bottom[0]->num();
        else
            num = bottom[0]->num() * bottom[0]->channels();
        
        int dim = bottom[0]->count() / num;
        
        // 计算均值，均值的形状是[num,1]
        caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1./dim, bottom_data,
                             sum_multiplier_.cpu_data(), 0, mean_.mutable_cpu_data());
        // 通过矩阵乘法，均值的形状转换为[num*dim]
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
                             mean_.cpu_data(), sum_multiplir_.cpu_data(), 0., 
                              temp_.mutable_cpu_data());
        // x-x_mean
        caffe_add(temp_.count(), bottom_data, temp_.cpu_data(), top_data);
        
        if(this->layer_param_.mvn_param().normalize_variance()){
            // 计算(x-x_mean)^2，方差形状转换为[num*dim]
            caffe_powx(bottom[0]->count(), top_data, Dtype(2), temp_.mutable_cpu_data());
            // 计算方差的平方，方差的形状为[num,1]
            caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1./dim, temp_.cpu_data(),
                                 sum_multiplier_.cpu_data(), 0., 
                                 variance_.mutable_cpu_data());
            // 开根号，计算方差
            caffe_powx(variance_.count(), variance_.cpu_data(), 
                       Dtype(0.5), variance_.mutable_cpu_data());
            // 保证数值安全
            caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
            // 通过矩阵乘法，方差的形状转换为[num*dim]
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., 
                                 variance_.cpu_data(), sum_multiplier_.cpu_data(), 0.,
                                 temp_.mutable_cpu_data());
            // 计算标准化值
            caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
        }
    }
    
    // ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    // ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    template <typename Dtype>
    void MVNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->cpu_diff();
        const Dtype* top_data = top[0]->cpu_data();
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        
        int num;
        if(this->layer_param_.mvn_param().across_channels())
            num = bottom[0]->num();
        else
            num = bottom[0]->num() * bottom[0]->channels();
        
        int dim = bottom[0]->count() / num;
        
        if(this->layer_param_.mvn_param().normalize_variance()){
            // 
            caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
            caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
                              sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., 
                                 mean_.cpu_data(), sum_multiplier_.cpu_data(),
                                 0, bottom_diff);
            caffe_mul(temp.count(), top_data, bottom_diff, bottom_diff);
            
            caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_diff, 
                          sum_multiplier_.cpu_data(), 0., mean_.mutable_cpu_data());
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., 
                                 mean_.cpu_data(), sum_multiplier_,cpu_data(),
                                 1., bottom_diff);
            caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1./dim),
                           bottom_diff);
            
            caffe_powx(temp_.count(), bottom_data, Dtype(2), temp_.mutable_cpu_data());
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1., 
                                  variance_.cpu_data(), sum_multiplier_.cpu_data(), 
                                  0., temp_.mutable_cpu_data());
            
            caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(),
                     bottom_diff);
        }
        else{
            caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1./dim, top_diff, 
                                  sum_multiplier_.cpu_data(), 0, 
                                  mean_.mutable_cpu_data());
            caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1,
                                 mean_.cpu_data(), sum_multiplier_.cpu_data(),
                                 0., temp_.mutable_cpu_data());
            caffe_add(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(MVNLayer);
#endif
    
    INSATNTIATE_CLASS(MVNLayer);
    REGISTER_LAYER_CLASS(MVN);
}

```

 1. caffe_cpu_gemv在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
        const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
        Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M, const int N, const float alpha, 		const float* A, const float* x, const float beta, float* y) {
    	cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    }
    
    template <>
    void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, 	const double* A, const double* x, const double beta, double* y) {
    	cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
    }
    ```

    cblas_sgemv和cblas_dgemv计算公式：C =alpha\*A\*b+beta\*C，函数定义如下，其中M和N是A的行数和列数，b和c的列数都是1：

    ```c++
    void cblas_sgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST float beta,  float  *y, OPENBLAS_CONST blasint incy);
    
    void cblas_dgemv(OPENBLAS_CONST enum CBLAS_ORDER order,  OPENBLAS_CONST enum CBLAS_TRANSPOSE trans,  OPENBLAS_CONST blasint m, OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double  *a, OPENBLAS_CONST blasint lda,  OPENBLAS_CONST double  *x, OPENBLAS_CONST blasint incx,  OPENBLAS_CONST double beta,  double  *y, OPENBLAS_CONST blasint incy);
    ```

 2. caffe_cpu_gemm在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, 	const int K, const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta, Dtype* C);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template<>
    void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const 	int N, const int K, const float alpha, const float* A, const float* B, const float beta,float* C) {
    	int lda = (TransA == CblasNoTrans) ? K : M;
    	int ldb = (TransB == CblasNoTrans) ? N : K;
    	cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
    }
    
    template<>
    void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, 	 	const int N, const int K, const double alpha, const double* A, const double* B, const double beta,
    	double* C) {
    	int lda = (TransA == CblasNoTrans) ? K : M;
    	int ldb = (TransB == CblasNoTrans) ? N : K;
    	cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, N);
    }
    ```

    cblas_sgemm和cblas_dgemm计算公式：C=alpha\*A\*B+beta\*C，A的形状是M*K，B的形状是K\*N, C的形状是M\*N:

    ```c++
    void cblas_sgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, 	 	OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, 	 	OPENBLAS_CONST blasint K, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *A, OPENBLAS_CONST blasint 	 lda, OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST float beta, float *C, 	 	OPENBLAS_CONST blasint ldc);
    
    void cblas_dgemm(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, 		OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, 		OPENBLAS_CONST blasint K, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *A, OPENBLAS_CONST 		blasint lda, OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, OPENBLAS_CONST double beta, 
        double *C, OPENBLAS_CONST blasint ldc);
    ```

 3. caffe_add在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_add_scalar(const int N, const float alpha, float* Y) {
    	for (int i = 0; i < N; ++i) {
    		Y[i] += alpha;
    	}
    }
    
    template <>
    void caffe_add_scalar(const int N, const double alpha, double* Y) {
    	for (int i = 0; i < N; ++i) {
    		Y[i] += alpha;
    	}
    }
    ```

 4. caffe_powx在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);
    ```

    caffe_powx在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_powx<float>(const int n, const float* a, const float b, float* y) {
    	vsPowx(n, a, b, y);
    }
    
    template <>
    void caffe_powx<double>(const int n, const double* a, const double b, double* y) {
    	vdPowx(n, a, b, y);
    }
    ```

    ```c++
    #define DEFINE_VSL_UNARY_FUNC_WITH_PARAM(name, operation) \
      template<typename Dtype> \
      void v##name(const int n, const Dtype* a, const Dtype b, Dtype* y) { \
        CHECK_GT(n, 0); CHECK(a); CHECK(y); \
        for (int i = 0; i < n; ++i) { operation; } \
      } \
      inline void vs##name( \
        const int n, const float* a, const float b, float* y) { \
        v##name<float>(n, a, b, y); \
      } \
      inline void vd##name( \
          const int n, const double* a, const float b, double* y) { \
        v##name<double>(n, a, b, y); \
      }
    
    DEFINE_VSL_UNARY_FUNC_WITH_PARAM(Powx, y[i] = pow(a[i], b))
    ```

 5. caffe_add_scalar在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cpp中进行实例化

    ```c++
    template <>
    void caffe_add_scalar(const int N, const float alpha, float* Y) {
    	for (int i = 0; i < N; ++i) {
    		Y[i] += alpha;
    	}
    }
    
    template <>
    void caffe_add_scalar(const int N, const double alpha, double* Y) {
    	for (int i = 0; i < N; ++i) {
    		Y[i] += alpha;
    	}
    }
    ```

 6. caffe_div在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cpp中进行实例化

    ```
    template <>
    void caffe_div<float>(const int n, const float* a, const float* b,float* y) {
    	vsDiv(n, a, b, y);
    }
    
    template <>
    void caffe_div<double>(const int n, const double* a, const double* b, double* y) {
    	vdDiv(n, a, b, y);
    }
    ```

    ```c++
    #define DEFINE_VSL_BINARY_FUNC(name, operation) \
      template<typename Dtype> \
      void v##name(const int n, const Dtype* a, const Dtype* b, Dtype* y) { \
        CHECK_GT(n, 0); CHECK(a); CHECK(b); CHECK(y); \
        for (int i = 0; i < n; ++i) { operation; } \
      } \
      inline void vs##name( \
        const int n, const float* a, const float* b, float* y) { \
        v##name<float>(n, a, b, y); \
      } \
      inline void vd##name( \
          const int n, const double* a, const double* b, double* y) { \
        v##name<double>(n, a, b, y); \
      }
    
    DEFINE_VSL_BINARY_FUNC(Add, y[i] = a[i] + b[i])
    DEFINE_VSL_BINARY_FUNC(Sub, y[i] = a[i] - b[i])
    DEFINE_VSL_BINARY_FUNC(Mul, y[i] = a[i] * b[i])
    DEFINE_VSL_BINARY_FUNC(Div, y[i] = a[i] / b[i])
    ```

    

**mvn_layer.cu**

```c++
#include <vector>

#include "caffe/layers/mvn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void MVNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
                                     const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        int num;
        if(this->layer_param_.mvn_param().across_channels())
            num = bottom[0]->num();
        else
            num = bottom[0]->num() * bottom[0]->channels();
        
        int dim = bottom[0]->count() / num();
        
        caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1./dim, bottom_data,
                             sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1,
                             mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                             temp_.mutable_gpu_data());
        caffe_gpu_add(temp_.count(), bottom_data, temp_.gpu_data(),
                     top_data);
        if(this->layer_param_.mvn_param().normalize_variance()){
            caffe_gpu_powx(bottom[0]->count(), top_data, Dtype(2),
                          temp_.mutable_gpu_data());
            caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1./dim, temp_.gpu_data(),
                                 sum_multiplier_.gpu_data(), 0., 
                                  variance_.mutable_gpu_data());
            
            caffe_gpu_powx(variance_.count(), variance_.gpu_data(), Dtype(0.5),
                           variance_.mutable_gpu_data());
            caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrance, num, dim, 1, 1.,
                                 variance_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                                  temp_.mutable_gpu_data());
            caffe_gpu_div(temp_.count(), top_data, temp_.gpu_data(), top_data);
        }
    }
    
    // ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    // ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    template <typename Dtype>
    void MVNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom){
        const Dtype* top_diff = top[0]->gpu_diff();
        const Dtype* top_data = top[0]->gpu_data();
        const Dtype* bottom_data = bottom[0]->gpu_diff();
        
        int num;
        if(this->layer_param_.mvn_param().across_channels())
            num = bottom[0]->num();
        else
            num = bottom[0]->num() * bottom[0]->channels();
        
        int dim = bottom[0]->count() / num;
        
        if(this->layer_param_.mvn_param().normalize_variance()){
            caffe_gpu_mul(temp_.count(), top_data, top_diff, bottom_diff);
            caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., bottom_diff,
                               sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                                 mean_.gpu_data(), sum_multiplier_.gpu_data(), 0.,
                                  bottom_diff);
            caffe_gpu_mul(temp_.count(), top_data, bottom_diff, bottom_diff);
            
            caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1, top_diff,
                               sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTransm, CblasNoTrans, num, dim, 1, 1.,
                                 mean_.gpu_data(), sum_multiplier_.gpu_data(), 1.,
                                 bottom_diff);
            caffe_gpu_axpby(temp_.count(), Dtype(1), top_diff, Dtype(-1./dim),
                           bottom_diff);
            
            caffe_gpu_powx(temp_.count(), bottom_data, Dtype(2), 		
                           	temp_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, 1.,
                                 variance_.gpu_data(), sum_multiplier_.gpu_data(),
                                  0., temp_.mutable_gpu_data());
            caffe_gpu_div(temp_.count(), bottom_diff, temp_.gpu_data(),
                         bottom_diff);
        }
        else{
            caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1./dim, top_diff,
                             sum_multiplier_.gpu_data(), 0., mean_.mutable_gpu_data());
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
                                 mean_.gpu_data(), sum_multiplier_.gpu_data(),
                                 0., temp_.mutable_gpu_data());
            caffe_gpu_add(temp_.count(), top_diff, temp_.gpu_data(), bottom_diff);
        }
    }
    
    INSTANTIATE_LAYER_GPU_FUCNS(MVNLayer);
}
```

 1. caffe_gpu_gemv在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N, const Dtype alpha, const 		Dtype* A, const Dtype* x, const Dtype beta, Dtype* y);
    ```

    caffe_gpu_gemv在math_functions.cu中进行实例化

    ```C++
    template <>
    void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    	const int N, const float alpha, const float* A, const float* x,
    	const float beta, float* y) {
    	cublasOperation_t cuTransA =
    		(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    	CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
    		A, N, x, 1, &beta, y, 1));
    }
    
    template <>
    void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    	const int N, const double alpha, const double* A, const double* x,
    	const double beta, double* y) {
    	cublasOperation_t cuTransA =
    		(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    	CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
    		A, N, x, 1, &beta, y, 1));
    }
    ```

 2. caffe_gpu_gemm在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
        const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
        const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
        Dtype* C);
    ```

    在math_functions.cu中进行实例化

    ```C++
    template <>
    void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    	const float alpha, const float* A, const float* B, const float beta, float* C) {
    	// Note that cublas follows fortran order.
    	int lda = (TransA == CblasNoTrans) ? K : M;
    	int ldb = (TransB == CblasNoTrans) ? N : K;
    	cublasOperation_t cuTransA =
    		(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    	cublasOperation_t cuTransB =
    		(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    	CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
    		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    }
    
    template <>
    void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    	const double alpha, const double* A, const double* B, const double beta, double* C) {
    	// Note that cublas follows fortran order.
    	int lda = (TransA == CblasNoTrans) ? K : M;
    	int ldb = (TransB == CblasNoTrans) ? N : K;
    	cublasOperation_t cuTransA =
    		(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    	cublasOperation_t cuTransB =
    		(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    	CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
    		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
    }
    ```

 3. caffe_gpu_add在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void add_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
    	CUDA_KERNEL_LOOP(index, n) {
    		y[index] = a[index] + b[index];
    	}
    }
    
    template <>
    void caffe_gpu_add<float>(const int N, const float* a, const float* b, float* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	add_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
    }
    
    template <>
    void caffe_gpu_add<double>(const int N, const double* a, const double* b, double* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	add_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
    }
    ```

 4. caffe_gpu_powx在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void powx_kernel(const int n, const Dtype* a, const Dtype alpha, Dtype* y) {
    	CUDA_KERNEL_LOOP(index, n) {
    		y[index] = pow(a[index], alpha);
    	}
    }
    
    template <>
    void caffe_gpu_powx<float>(const int N, const float* a, const float alpha, float* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	powx_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, alpha, y);
    }
    
    template <>
    void caffe_gpu_powx<double>(const int N, const double* a, const double alpha, double* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	powx_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, alpha, y);
    }
    ```

 5. caffe_gpu_add_scalar在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
    	CUDA_KERNEL_LOOP(index, n) {
    		y[index] += alpha;
    	}
    }
    
    template <>
    void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	add_scalar_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, alpha, Y);
    }
    
    template <>
    void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	add_scalar_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, alpha, Y);
    }
    ```

 6. caffe_gpu_div在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void div_kernel(const int n, const Dtype* a, const Dtype* b, Dtype* y) {
    	CUDA_KERNEL_LOOP(index, n) {
    		y[index] = a[index] / b[index];
    	}
    }
    
    template <>
    void caffe_gpu_div<float>(const int N, const float* a, const float* b, float* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	div_kernel<float> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
    }
    
    template <>
    void caffe_gpu_div<double>(const int N, const double* a, const double* b, double* y) {
    	// NOLINT_NEXT_LINE(whitespace/operators)
    	div_kernel<double> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> > (N, a, b, y);
    }
    ```

    
