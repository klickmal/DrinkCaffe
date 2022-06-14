# Tile Layer
tile layer将数据沿着指定的轴进行复制。

tile layer在源文件tile_layer.hpp, tile_layer.cpp和tile_layer.cu中实现

**tile_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
// 将数据按照某个维度扩大n倍，将bottom_data的前inner_dim_个数据复制了tiles份，
// 反向时将对应diff累加回去即可
    
	template <typename Dtype>
	class TileLayer: public Layer<Dtype>{
    public:
        explicit TileLayer(const LayerParameter& param):
            Layer<Dtype>(param){}
    	
    	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    
        virtual inline const char* type() const{return "Tile";}
    	virtual inline const int ExactNumBottomBlobs() const {return 1;}
     	virtual inline const int ExactNumTopBlobs() const {return 1;}

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
    	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype*>& bottom);
    	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                                  const vector<bool>& propagate_down,
                                 const vector<Blob<Dtype>*>& bottom);
    
    	unsigned int axis_ /*在指定轴复制*/, tiles_ /*复制的份数*/, outer_dim_, inner_dim_;
    };
}
```

**tile_layer.cpp**

```c++
#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    void TileLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top){
        const TileParamter& tile_param = this->layer_param_.tile_param();
        axis_ = bottom[0]->CanonicalAxisIndex(tile_param.axis());
        CHECK(tile_param.has_tiles()) << "Number of tiles must be specified";
        
        tiles_ = tile_param.tiles();
        CHECK_GT(tiles, 0) << "Number of tiles must be positive";
        
        // 计算top blob的shape
        vector<int> top_shape = bottom[0]->shape();
        top_shape[axis_] = bottom[0]->shape(axis_)*tiles_;
        
        top[0]->Reshape(top_shape);
        outer_dim_ = bottom[0]->count(0, axis_);
        inner_dim = bottom[0]->count(axis_);
    }
    
    // 将数据按照某个维度扩大n倍，看下面forward源码，将bottom_data的前inner_dim_个数据复制了tiles份
    template <typename Dtype>
    void TileLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top){
       const Dtype* bottom_data = bottom[0]->cpu_data();
       Dtype* top_data = top[0]->mutable_cpu_data();
       
       for(int i=0; i<outer_dim_; ++i){
           for(int t=0; t<tile_; ++t){
               caffe_copy(inner_dim_, bottom_data, top_data);
               top_data += inner_dim_;
           }
           bottom_data += inner_dim_;
       }
    }
    
    template <typename Dtype>
    void TileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_diff = top[0]->cpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
            
            for(int i=0; i<outer_dim_; ++i){
                // 赋值一份梯度到bottom
                caffe_copy(inner_dim_, top_diff, bottom_diff);
                
                top_diff += inner_dim_;
                for(int t=1;t<tiles_; ++t){
                    caffe_axpy(inner_dim_, Dtype(1), top_diff, bottom_diff);
                    top_diff += inner_dim_;
                }
                bottom_diff += inner_dim_;
            }
        }
    }

#ifdef CPU_ONLY
	STUP_GPU(TileLayer);
#endif
    
    INSTANTIATE_CLASS(TileLayer);
    REGISTER_LAYER_CLASS(Tile);
}

```

 1. caffe_axpy函数在math_functions.hpp中声明：

    ```c++
    template <typename Dtype>
    void caffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);
    ```

    caffe_axpy在math_functions.cpp实现如下：

    ```c++
    template <>
    void caffe_axpy<float>(const int N, const float alpha, const float* X, float* Y) {
    	cblas_saxpy(N, alpha, X, 1, Y, 1);
    }
    
    template <>
    void caffe_axpy<double>(const int N, const double alpha, const double* X, double* Y) {
    	cblas_daxpy(N, alpha, X, 1, Y, 1);
    }
    ```

    cblas_saxpy、cblas_daxpy在OpenBLAS接口定义如下，其功能如下式：
    $$
    Y=alpha * X +beta*Y
    $$

    ```c++
    void cblas_saxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST float alpha, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
    void cblas_daxpy(OPENBLAS_CONST blasint n, OPENBLAS_CONST double alpha, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
    ```

**tile_layer.cu**

```c++
#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	
    // 怎么理解index呢？？？？？？？？？？
    // tile_size = c*h*w 
    // num_tiles = t
    // bottom_tile_axis = c
    // d是c*h*w的index
    // b是
    template <typename Dtype>
    __global__ void Tile(const int nthreads, const Dtype* bottom_data, 
                    const int tile_size, const int num_tiles, 
                    const int bottom_tile_axis, Dtype* top_data){
		CUDA_KERNEL_LOOP(index, nthreads){
            const int d = index % tile_size;
            // 因为index在某一个轴上多了num_tiles份，所以多除一个num_tiles
            const int b = (index/tile_size/num_tiles)%bottom_tile_axis;
            const int n = index/tile_size/num_tiles/bottom_tile_axis;
            const int bottom_index = (n*bottom_tile_axis+b)*tile_size + d;
            
            top_index[index] = bottom_data[bottom_index];
        }
    }
    
    // 假如bottom blob的shape是[n,c,h,w], 指定tile的axis=1，tile成[n,ct,h,w]
    // 那么
    // bottom_tile_axis = c
    // nthreads = n*ct*h*w
    template <typename Dtype>
    void TileLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top){
        const Dtype* bottom_data = bottom[0]->gpu_data();
        Dtype* top_data = top[0]->mutable_gpu_data();
        const int bottom_tile_axis = bottom[0]->shape(axis_);
        
        const int nthreads = top[0]->count();
        
        Tile<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads,
                        bottom_data, inner_dim_, tiles_, bottom_tile_axis, top_data);
    }
    
    // tile_size = h*w
    // num_tiles = t
    // bottom_tile_axis = c
    template <typename Dtype>
    __global__ void TileBackward(const int nthreads, const Dtype* top_diff,
                                const int tile_size, const int num_tiles, 
                                const int bottom_tile_axis, Dtype* bottom_diff){
        CUDA_KERNERL_LOOP(index, nthreads){
            // d是height*width的index
            // b是channel的index
            // n是batch的index
            const int d = index % tile_size;
            const int b = (index/tile_size)%bottom_tile_axis;
            const int n = index/tile_size/bottom_tile_axis;
            bottom_diff[index] = 0;
            int top_index = (n*num_tiles*bottom_tile_axis+b)*tile_size+d;
            for(int t=0; t<num_tiles; ++t){
                bottom_diff[index] += top_diff[top_index];
                top_index += bottom_tile_axis*tile_size;
            }
        }
    }
    
    // 假如bottom blob的shape是[n,c,h,w], 指定tile的axis=1，tile成[n,ct,h,w]
    // 那么
    // bottom_tile_axis = c
    // tile_size = c*h*w/c = h*w
    // nthreads = n*c*h*w
    template <typename Dtype>
    void TileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom){
        if(propagate_down[0]){
            const Dtype* top_diff = top[0]->gpu_diff();
            Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
            const int bottom_tile_axis = bottom[0]->shape(axis_);
            const int tile_size = inner_dim_/bottom_tile_axis;
            const int nthreads = bottom[0]->count();
            
            TileBackward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
            	nthreads, top_diff, tile_size, tiles_, bottom_tile_axis, bottom_diff); 
        }
    }
    INSTANTIATE_LAYER_GPU_FUNCS(TileLayer);
}
```

