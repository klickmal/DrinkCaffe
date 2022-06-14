# Silence Layer
silence layer忽略bottom blobs并且不产生top blobs

**silence_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

template <typename Dtype>
class SilenceLayer: public Layer<Dtype>{
    public:
        explicit SilenceLayer(const LayerParameter& param):
            Layer<Dtype>(param){}
    	
    	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top){}
    	
        virtual inline const char* type() const{return "Silence";}
    	virtual inline int MinBottomBlobs() const {return 1;}
    	virtual inline int ExactNumTopBlobs() const {retun 1;}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top){}
    	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype*>& top);
    	virtual void Backward_gpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& top);
    };
}
```

**silence_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void SilenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                             const vector<bool>& propagate_down,
                                             const vector<Blob<Dtype>*>& bottom){
       for(int i=0; i<bottom.size(); ++i){
           if(propagate_down[i]){
               caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_cpu_diff());
           }
       }
    }

#ifdef CPU_ONLY
	STUP_GPU(SilenceLayer);
#endif
    
    INSTANTIATE_CLASS(SilenceLayer);
    REGISTER_LAYER_CLASS(Silence);
}

```

 1. caffe_set在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_set(const int N, const Dtype alpha, Dtype *X);
    ```

    caffe_set在math_functions.cpp实现如下：

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
    
    template void caffe_set<int>(const int N, const int alpha, int* Y);
    template void caffe_set<float>(const int N, const float alpha, float* Y);
    template void caffe_set<double>(const int N, const double alpha, double* Y);
    ```

**silence_layer.cu**

```c++
#include <vector>

#include "caffe/layers/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void SilenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top){
        // Do Nothing
    }
    
    template <typename Dtype>
    void SilenceLayer<Dtype>::Backeard_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom){
        for(int i=0; i<bottom.size(); ++i){
            if(propagate_down[i]){
                caffe_gpu_set(bottom[i]->count(), Dtype(0), 
                              bottom[i]->mutable_gpu_diff());
            }
        }
    }
    
    INSTANTIATE_LAUER_GPU_FUNCS(SilenceLayer);
}
```

 1. caffe_gpu_set在math_functions.hpp中声明如下：

    ```c++
    template <typename Dtype>
    void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);
    ```

    在math_functions.cu中进行实例化

    ```c++
    template <typename Dtype>
    __global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
      CUDA_KERNEL_LOOP(index, n) {
        y[index] = alpha;
      }
    }
    
    template <typename Dtype>
    void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
      if (alpha == 0) {
        CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
        return;
      }
      // NOLINT_NEXT_LINE(whitespace/operators)
      set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
          N, alpha, Y);
    }
    
    template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
    template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
    template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);
    ```

    
