# Fillers
Fillers类在filler.hpp中实现
**filler.hpp**

```c++
#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"s

namespace caffe{
    
    template <typename Dtype>
    class Filler{
    public:
        explicit Filler(const FillerParameter& param) : filler_param_(param) {}
        virtual ~Filler() {}
        virtual void Fill(Blob<Dtype>* blob) = 0;
    protected:
        FillerParameter filler_param_;
    };
    
    // 常量初始化类
    template <typename Dtype>
    class ConstantFiller: public Filler<Dtype>{
    public:
        explicit ConstantFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            Dtype* data = blob->mutable_cpu_data();
            const int count = blob->count();
            const Dtype value = this->filler_param_.value();
            CHECK(count);
            for(int i=0; i<count; ++i){
                data[i] = value;
            }
            
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
    
    // 均匀分布初始化类
    template <typename Dtype>
    class UniformFiller: public Filler<Dtype>{
    public:
        explicit UniformFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            CHECK(blob->count());
            caffe_rng_uniform<Dtype>(blob->count(), Dtype(this->filler_param_.min()),
                       Dtype(this->filler_param_.max(), blob->mutable_cpu_data());
            CHECK_EQ(this->filler_param_.sparse(), -1)
                       << "Sparsity not supported by this Filler.";
        }
    };
    
    // 高斯分布初始化类（支持稀疏特性）
    template <typename Dtype>
    class GaussianFiller: public Filler<Dtype>{
    public:
        explicit GaussianFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            Dtype* data = blob->mutable_cpu_data();
            CHECK(blob->count());
            caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()),
                        Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
            int sparse = this->filler_param_.sparse();
            CHECK_GE(sparse, -1);
            if(sparse >= 0){
                CHECK_GE(blob->num_axes(), 1);
                const int num_outputs = blob->shape(0);
                Dtype non_zero_probability = Dtype(sparse) / Dtype(num_outputs);
                rand_vec_.reset(new SyncedMemory(blob->count() * sizeof(int)));
                int* mask = reinterpret_cast<int*>(rand_vec_->mutable_cpu_data());
                caffe_rng_bernoulli(blob->count(), non_zero_probability, mask);
                for(int i=0; i<blob->count(); ++i){
                    data[i] *= mask[i];
                }
            }
        }
     protected:
        shared_ptr<SyncedMemory> rand_vec_;
    };
    
    // 略过
    template <typename Dtype>
   	class PositiveUnitBallFiller: public Filler<Dtype>{
    public:
        explicit PositiveUnitballFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            Dtype* data = blob->mutable_cpu_data();
            DCHECK(blob->count());
            caffe_rng_uniform<Dtype>(blob->count(), 0, 1, blob->mutable_cpu_data());
            
            int dim = blob->count() / blob->num();
            CHECK(dim);
            for(int i=0; i<blob->num(); ++i){
                Dtype sum=0;
                for(int j=0; j<dim; ++j){
                    sum += data[i*dim + j];
                }
                for(int j=0; j<dim; ++j){
                    data[i*dim+j] /=sum;
                }
            }
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
    
    // XavierFiller初始化(用于卷积核): Uniform for [-sqrt(3/n), sqrt(3/n)]
	// 扇入扇出的定义：(num, a, b, c) where a * b * c = fan_in and num * b * c = fan_out     
    template <typename Dtype>
    class XavierFiller: public Filler<Dtype>{
    public:
        explicit XavierFiller(const FillerParameter& param)
            : Filler<Dtype>(param){}
        virtual void Fill(Blob<Dtype>* blob){
            CHECK(blob->count());
            int fan_in = blob->count()/blob->num();
            int fan_out = blob->count()/blob->channels();
            Dtype n = fan_in;
            if(this->filler_param_.variance_norm() == 
              FillerParameter_VarianceNorm_AVERAGE){
                n = (fan_in + fan_out)/Dtype(2);
            }
            else if(this->filler_param_.variance_norm() ==
                   FillerParameter_VarianceNorm_FAN_OUT){
                n = fan_out;
            }
            Dtype scale = sqrt(Dtype(3)/n);
            caffe_rng_uniform<Dtype>(blob->count(), -scale, scale,
                                    blob->mutable_cpu_data());
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
    
    // MSRAFiller初始化方式(用于卷积核)
    // MSRA的基本思想与Xavier类似，但主要是针对ReLU和PReLU激活函数来设计的。
    // 这种方法在实际应用时根据一个方差为sqrt(2/n),均值为0的高斯分布来初始化权值
    // https://blog.csdn.net/caicaiatnbu/article/details/100750076
    template <typanema Dtype>
    class MSRAFiller: public Filler<Dtype>{
    public:
        explicit MSRAFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            CHECK(blob->count());
            int fan_in = blob->count()/blob->num();
            int fan_out = blob->count()/blob->channels();
            Dtype n = fan_out;
            if(this->filler_param_.variance_norm() == 
              FillerParameter_VarianceNorm_AVERAGE){
                n = (fan_in + fan_out) / Dtype(2);
            }
            else if(this->filler_param_.variance_norm() ==
                   FillerParameter_VarianceNorm_FAN_IN){
                n = fan_in;
            }
            Dtype alpha = this->filler_param_.alpha();
            Dtype std = sqrt(Dtype(2) / n / (1 + alpha*alpha));
            caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std,
                                     blob->mutable_cpu_data());
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
    
    // BilinearFiller初始化(用于反卷积核)
    template <typename Dtype>
   	class BilinearFiller: public Filler<Dtype>{
    public:
        explicit BilinearFiller(const FillerParameter& param)
            : Filler<Dtype>(param){}
        virtual void Fill(Blob<Dtype>* blob){
            CHECK_EQ(blob->num_axes(), 4) << "Blob must be 4 dim.";
            CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
            Dtype* data = blob->mutable_cpu_data();
            int f = ceil(blob->width() / 2.);
            float c =(2*f-1-f%2)/(2.*f);
            for(int i=0; i<blob->count(); ++i){
                float x = i%blob->width();
                float y = (i/blob->width()) % blob->height();
                data[i] = (1-fabs(x/f-c))*(1-fabs(y/f-c));
            }
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
    
    // 略过
    template <typename Dtype>
  	class IdentityFiller: public Filler<Dtype>{
    public:
        explicit IdentityFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            CHECK(blob->count());
            int fan_in = blob->count() / blob->num();
            int fan_out = blob->count() / blob->channels();
            CHECK_EQ(fan_in, fan_out);
            Dtype* blob_data = blob->mutable_cpu_data();
            caffe_set(blob->count(), Dtype(0), blob_data);
            for(int i=0; i<blob->num(); ++i){
                blob_data[i*blob->channels()+i] = Dtype(1);
            }
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
    
    // 略过
    template <typename Dtype>
    class GaussianUnitBallFiller: public Filler<Dtype>{
    public:
        explicit GaussianUnitBallFiller(const FillerParameter& param)
            : Filler<Dtype>(param) {}
        virtual void Fill(Blob<Dtype>* blob){
            CHECK(blob->count());
            int fan_in = blob->count() / blob->num();
            int n = fan_in;
            caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), 1, 
                                      blob->mutable_cpu_data());
            Dtype sum_sq;
            for(int i=0; i<blob->num(); i++){
                sum_sq = caffe_cpu_dot(n, blob->cpu_data() + i*n, 
                                       blob->cpu_data() + i*n) + 1e-12;
                caffe_cpu_scale<Dtype>(n, Dtype(1.0)/sqrt(sum_sq), blob->cpu_data()+i*n,
                                      blob->mutable_cpu_data() + i*n);
            }
            CHECK_EQ(this->filler_param_.sparse(), -1)
                << "Sparsity not supported by this Filler.";
        }
    };
     
    template <typename Dtype>
    Filler<Dtype>* GetFiller(const FillerParameter& param){
        const std::string& type = param.type();
        if(type == "constant"){
            return new ConstantFiller<Dtype>(param);
        }
        else if(type=="gaussian"){
            return new GaussianFiller<Dtype>(param);
        }
        else if(type=="positive_unitball"){
            return new PositiveUnitballFiller<Dtype>(param);
        }
        else if(type=="uniform"){
            return new UniformFiller<Dtype>(param);
        }
        else if(type=="xavier"){
            return new XavierFiller<Dtype>(param);
        }
        else if(type=="msra"){
            return new MSRAFiller<Dtype>(param);
        }
        else if(type=="bilinear"){
            return new BilinearFiller<Dtype>(param);
        }
        else if(type=="identity"){
            return new IdentityFiller<Dtype>(param);
        }
        else if(type=="gaussian_unitball"){
            return GaussianUnitBallFiller<Dtype>(param);
        }
        else{
            CHECK(false) << "Unkonw filler name: " << param.type();
        }
        return (Filler<Dtype>*)(NULL);
    }
}
```





