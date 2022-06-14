# Blob
Blob类：

* A wrapper around SyncedMemory holders serving as the basic computational unit through which Layer, Net, and Solver

​		interact.

Blob类在blob.hpp和blob.cpp中实现

**blob.hpp**

```c++
#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe{
    
    template <typename Dtype>
    class Blob{
    public:
        Blob(): data_(), diff(), count_(0), capacity_(0){}
        
        explicit Blob(const int num, const int channels, 
                      const int height, const int width);
        explicit Blob(const vector<int>& shape);
        
        // Reshape函数，必要时可以重新分配内存
        bool Reshape(const int num, const int channels, 
                     const int height, const int width);
        
        bool Reshape(const vector<int>& shape);
        bool Reshape(const BlobShape& shape);
        bool ReshapeLike(const Blob& other);
        
        // 打印shape信息，如：1 2 3 4 (24)
        inline string shape_string() const{
            ostringstream stream;
            for(int i=0; i < shape_.size(); ++i){
                stream << shape_[i] << " ";
            }
            stream << "(" << count_ << ")";
            return stream.str();
        }
        
        inline const vector<int>& shape() const {return shape_;}
        inline int shape(int index) const {
            return shape_[CanonicalAxisIndex(index)];
        }
        inline int num_axes() const {return shape_.size();}
        inline int count() const {return count_;}
        
        inline int count(int start_axis, int end_axis) const {
            CHECK_LE(start_axis, end_axis);
            CHECK_GE(start_axis, 0);
            CHECK_GE(end_axis, 0);
            CHECK_LE(start_axis, num_axes());
            CHECK_LE(end_axis, num_axes());
            int count = 1;
            for(int i = start_axis; i < end_axis; ++i){
                count *= shape(i);
            }
            return count;
        }
        
        // 负索引转换函数
        inline int CanonicalAxisIndex(int axis_index) const {
            CHECK_GE(axis_index,  -num_axes())
                << "axis " << axis_index << "out of range for " << num_axes()
                << "-D Blob with shape " << shape_string();
            CHECK_LT(axis_index, num_axes())
                << "axis " << axis_index << "out of range for " << num_axes()
                << "-D Blob with shape " << shape_string();
            if(axis_index < 0){
                return axis_index + num_axes();
            }
            return axis_index;
        }
        
        inline int num() const {return LegacyShape(0);}
        inline int channels() const {return LegacyShape(1);}
        inline int height() const {return LegacyShape(2);}
        inline int width() const {return LegacyShape(3);}
        
        inline int LegacyShape(int index) const{
            CHECK_LE(num_axes(), 4)
                << "Cannot use legacy accessors on Blobs with  > 4 axes. ";
            CHECK_LT(index, 4);
            CHECK_GE(index, -4);
            if(index >= num_axes() || index < -num_axes()){
                return 1;
            }
            return shape(index);
        }
        
        // 计算元素偏移量
        inline int offset(const int n, const int c = 0; const int h = 0; const int w = 0)
            const{
            CHECK_GE(n, 0);
            CHECK_LE(n, num());
            CHECK_GE(channels(), 0);
            CHECK_LE(c, channels());
            CHECK_GE(height(), 0);
            CHECK_LE(h, height());
            CHECK_GE(width(), 0);
            CHECK_LE(w, width());
            
            return ((n * channles() + c) * height() + h) * width() + w;
        }
        
        inline int offset(const vector<int>& indices) const {
            CHECK_LE(indices.size(), num_axes());
            int offset = 0;
            for(int i = 0; i < num_axes(); ++i){
                offset *= shape(i);
                if(indices.size() > 1){
                    CHECK_GE(indices[i], 0);
                    CHECK_LT(indices[i], shape(i));
                    offset += indices[i];
                }
            }
            return offset;
        }
        
        void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false, 
                      bool reshape = false);
         
        inline Dtype data_at(const int n, const int c, const int h, const int w) const{
            return cpu_data()[offset(n, c, h, w)];
        }
        
        inline Dtype diff_at(const int n, const int c, const int h, const int w) const{
            return cpu_diff()[offset(n, c, h, w)];
        }
        
        inline Dtype data_at(const vector<int>& index) const {
            return cpu_data()[offset(index)];
        }
        
        inline Dtype diff_at(const vector<int>& index) const {
            return cpu_diff()[offset(index)];
        }
        
        inline const shared_ptr<SyncedMemory>* data() const{
            CHECK(data_);
            return data_;
        }
        
        inline const shared_ptr<SyncedMemory>* diff() const{
            CHECK(diff_);
            return diff_;
        }
        
        const Dtype* cpu_data() const;
        void set_cpu_data(Dtype* data);
        void set_cpu_diff(Dtype* data);
        void set_gpu_data(Dtype* data);
        void set_gpu_diff(Dtype* data);
        const int* gpu_shape() const;
        const Dtype* gpu_data() const;
        const Dtype* cpu_diff() const;
        const Dtype* gpu_diff() const;
        
        Dtype* mutable_cpu_data();
        Dtype* mutable_gpu_data();
        Dtype* mutable_cpu_diff();
        Dtype* mutable_gpu_diff();
        
        // 根据梯度更新data_: x = x - learning_rate * g(x)
        void Update();
        // 反序列化，从BlobProto中恢复Blob
        void FromProto(const BlobProto& proto, bool reshape = true);
        // 序列化，将内存中的Blob对象保存到BlobProto中
        void ToProto(BlobProto* proto, bool write_diff = false) const;
        
        // Compute the sum of absolute values (L1 norm) of the data.
        Dtype asum_data() const;
        // Compute the sum of absolute values (L1 norm) of the diff
        Dtype asum_diff() const;
        // Compute the sum of squares (L2 norm squared) of the data
        Dtype sumsq_data() const;
        // Compute the sum of squares (L2 norm squared) of the diff
        Dtype sumsq_diff() const;
        
        void Clamp(Dtype lower_bound, Dtype upper_bound);
        void scale_data(Dtype scale_factor);
        void scale_diff(Dtype scale_factor);
        
        void ShareData(const Blob& other);
        void ShareDiff(const Blob& other);
        bool ShareEquals(const BlobProto& other);
        
    protected:
        shared_ptr<SyncedMemory> data_;
        shared_ptr<SyncedMemory> diff_;
        shared_ptr<SyncedMemory> shape_data_; //存储blob数据空间的维度，比如图像数据N、C、H、W。
        vector<int> shape_;
        int count_; // Blob的数据大小，即shape_各个元素相乘。比如图像数据大小为N*C*H*W
        int capacity_; // Blob的数据内存空间容量capacity_，必须大于等于count_
        
        DISABLE_COPY_AND_ASSIGN(Blob);
    };
}
```

**blob.cpp**

```c++
#include <climits>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
    
    template <typename Dtype>
    bool Blob<Dtype>::Reshape(const int num, const int channels, const int height,
                             const int width){
        vector<int> shape(4);
        shape[0] = num;
        shape[1] = channels;
        shape[2] = height;
        shape[3] = width;
        
        return Reshape(shape);
    }
    
    template <typename Dtype>
    bool Blob<Dtype>::Reshape(const vector<int>& shape){
        CHECK_LE(shape.size(), kMaxBlobAxes);
        count_ = 1;
        shape_.resize(shape.size());
        
        if(!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)){
            shape_data_.reset(new SyncedMemory(shape.size() * size(int)));
        }
        
        int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
        for(int i = 0; i < shape.size(); ++i){
            CHECK_GE(shape[i], 0);
            if(count_ != 0){
                CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
            }
            count_ *= shape[i];
            shape_[i] = shape[i];
            shape_data[i] = shape[i];
        }
        if(count_ > capacity_){
            capacity_ = count_;
            data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
            diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
            return true;
        }
        else{
            return false;
        }
    }
    
    template <typename Dtype>
    bool Blob<Dtype>::Reshape(const BlobShape& shape){
        CHECK_LE(shape.dim_size(), kMaxBlobAxes);
        vector<int> shape_vec(shape.dim_size());
        for(int i = 0; i < shape.dim_size(); ++i){
            shape_vec[i] = shape.dim(i);
        }
        return Reshape(shape_vec);
    }
    
    template <typename Dtype>
    bool Blob<Dtype>::Blob(const int num, const int channels, const int height,
                          const int width): capacity_(0){
        Reshape(num, channels, height, width);
    }
    
    template <typename Dtype>
    bool Blob<Dtype>::Blob(const vector<int>& shape): capacity_(0){
        Reshape(shape);
    }
    
    template <typename Dtype>
    const int* Blob<Dtype>::gpu_shape() const {
        CHECK(shape_data_);
        return (const int*)shape_data_->gpu_data();
    }
    
    template <typename Dtype>
    const Dtype* Blob<Dtype>:;cpu_data() const {
        CHECK(data_);
        return (const Dtype*)data_->cpu_data();
    }
    
    template <typename Dtype>
    void Blob<Dtype>::set_cpu_data(Dtype* data){
        CHECK(data);
        size_t size = count_ * sizeof(Dtype);
        if(data_->size() != size){
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        data_->set_cpu_data(data);
    }
    
    template <typename Dtype>
    void Blob<Dtype>::set_cpu_diff(Dtype* diff){
        CHECK(diff);
        diff_->set_cpu_data(diff);
    }
    
    template <typename Dtype>
    void Blob<Dtype>::set_gpu_diff(Dtype* diff){
        CHECK(diff);
        diff_->set_gpu_data(diff);
    }
    
    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_data() const {
        CHECK(data_);
        return (const Dtype*)data_->gpu_data();
    }
    
    template <typename Dtype>
    void Blob<Dtype>::set_gpu_data(Dtype* data){
        CHECK(data);
        size_t size = count_ * sizeof(Dtype);
        if(data_->size() != size){
            data_.reset(new SyncedMemory(size));
            diff_.reset(new SyncedMemory(size));
        }
        data_->set_gpu_data(data);
    }
    
    template <typename Dtype>
    const Dtype* Blob<Dtype>::cpu_diff() const{
        CHECK(diff_);
        return (const Dtype*)diff_->cpu_data();
    }
    
    template <typename Dtype>
    const Dtype* Blob<Dtype>::gpu_diff() const{
        CHECK(diff_);
        return (const Dtype*)diff_->gpu_data();
    }
    
    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cpu_data() {
        CHECK(data_);
        return static_cast<Dtype*>(data_->mutable_cpu_data());
    }
    
    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_data() {
        CHECK(data_);
        return static_cast<Dtype*>(data_->mutable_gpu_data());
    }
    
    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_cou_diff(){
        CHECK(diff_);
        return static_cast<Dtype*>(diff_->mutable_cpu_data());
    }
    
    template <typename Dtype>
    Dtype* Blob<Dtype>::mutable_gpu_diff(){
        CHECK(diff_);
        return static_cast<Dtype*>(diff_->mutable_gpu_data());
    }
    
    template <typename Dtype>
    void Blob<Dtype>::ShareData(const Blob& other){
        CHECK_EQ(count_, other.count());
        data_ = other.data();
    }
    
    template <typename Dtype>
    void Blob<Dtype>::ShareDiff(const Blob& other){
        CHECK_EQ(count_, other.count());
        diff_ = other.diff();
    }
    
    template <> void Blob<unsigned int>::Updata() {NOT_IMPLEMENTED;}
    template <> void Blob<int>::Updata() {NOT_IMPLEMENTED;}
    
    template <typename Dtype>
    void Blob<Dtype>::Updata(){
        // 根据数据在cpu还是gpu，决定在哪更新
        switch(data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                // 总之其原理就是:  x = x -  tidu(x)
                // 学习速率在Back_Forward函数里面作为梯度的一部分算在diff_里面了，
                // 所以这个caffe_axpy第二个参数是-1。后面会提到，这个要注意。
                caffe_axpy<Dtype>(count, Dtype(-1), 
                                  static_cast<const Dtype*>(diff_->cpu_data()),
                                  static_cat<const Dtype*>(data_->mutable_cpu_data()));
                break;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY:
                caffe_gpu_axpy<Dtype>(count_, Dtype(-1),
                                     static_cast<const Dtype*>(diff_->gpu_data()),
                                     static_cast<Dtype*>(data_->mutable_cpu_data()));
#else:
                NO_GPU;
#endif
                break;
            default:
                LOG(FATAL) << "Syncedmem not initialized.";
        }
    } 
    
    template <> unsigned int Blob<unsigned int>::asum_data() const{
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <> int Blob<int>::asum_data() const{
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::asum_data() const{
        if(!data_) {return 0;}
        switch(data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                return caffe_cpu_asum(count_, cpu_data());
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifdef CPU_ONLY
                {
                    Dtype asum;
                    caffe_gpu_asum(count_, gpu_data(), &asum);
                    return asum;
                }
#else
                NO_GPU;
#endif
            case SyncedMemory::UNINITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
        return 0;
    }
    
    template <> unsigned int Blob<unsigned int>::asum_diff() const{
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <> int Blob<int>::asum_diff() const {
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::asum_diff() const{
        if(!diff) {return 0;}
        switch(diff_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                return caffe_cpu_asum(count_, cpu_diff());
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY:
                {
                    Dtype asum;
                    caffe_gpu_asum(count_, gpu_diff(), &asum);
                    return asum;
                }
#else
                NO_GPU;
#endif
            case SyncedMemory::UNINITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
        }
        return 0;
    }
    
    template <> unsigned int Blob<unsigned int>::sumsq_data() const{
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <> int Blob<int>::sumsq_data() const {
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::sumsq_data() const{
        Dtype sumsq;
        const Dtype* data;
        if(!data_) {return 0};
        switch(data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
            	data = cpu_data();
                sumsq = caffe_cpu_dot(count_, data, data);
                break;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY:
                data = gpu_data();
                caffe_gpu_dot(count_, data, data, &sumsq);
#else
                NP_GPU;
#endif
                break;
            case SyncedMemory::UNINTIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
        return sumsq;
    }
    
    template <> unsigned int Blob<unsigned int>::sumsq_diff() const{
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <> int Blob<int>::sumsq_diff() const{
        NOT_IMPLEMENTED;
        return 0;
    }
    
    template <typename Dtype>
    Dtype Blob<Dtype>::sumsq_diff() const{
        Dtype sumsq;
        const Dtype* diff;
        if(!diff_) {return 0;}
        switch(diff_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                diff = cpu_diff();
                sumsq = caffe_cpu_dot(count_, diff, diff);
                break;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
                diff = gpu_diff();
                caffe_gpu_dot(count_, diff, diff, &sumsq);
                break;
#else
                NO_GPU;
#endif
            case SyncedMemory::UNITIALIZED:
                return 0;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
        return sumsq;
    }
    
    template <> void Blob<unsigned int>::scale_data(unsigned int scale_factor){
        NOT_IMPLEMENTED;
    }
    
    template <> void Blob<int>::scale_data(int scale_factor){
        NOT_IMPLEMENTED;
    }
    
    template <typename Dtype>
    void Blob<Dtype>::scale_data(Dtype scale_factor){
        Dtype* data;
        if(!data_) {return;}
        
        switch(data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                data = mutable_cpu_data();
                caffe_scal(count_, scale_factor, data);
              	return;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
                data = mutable_gpu_data();
                caffe_gpu_scal(count_, scale_factor, data);
                return;
#else
                NO_GPU;
#endif
            case SyncedMemory::UNITIALIZED:
                return;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
    }
    
    template <> void Blob<unsigned int>::scale_diff(unsigned int scale_factor){
        NOT_IMPLEMENTED;
    }
    
    template <>  void Blob<int>::scale_diff(int scale_factor){
        NOT_IMPLEMENTED;
    }
    
    template <typename Dtype>
    void Blob<Dtype>::scale_diff(Dtype scale_factor){
        Dtype* diff;
        if(!diff_) {return;}
        switch(diff_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                diff = mutable_cpu_diff();
                caffe_scal(count_, scale_factor, diff);
                return;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
                diff = mutable_gpu_diff();
                caffe_gpu_scal(count_, scale_factor, diff);
                return;
#else
                NO_GPU;
#endif
            case SyncedMemory::UNITIALIZED:
                return;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << diff_->head();
        }
    }
    
    template <typename Dtype>
    void Blob<Dtype>::Clamp(Dtype lower_bound, Dtype upper_bound){
        Dtype* data;
        if(!data_) {return;}
        
        switch(data_->head()){
            case SyncedMemory::HEAD_AT_CPU:
                data = mutable_cpu_data();
                for(int i = 0; i < count_; ++i){
                    data[i] = std::min(std::max(data[i], lower_bound), upper_bound);
                }
                return;
            case SyncedMemory::HEAD_AT_GPU:
            case SyncedMemory::SYNCED:
#ifndef CPU_ONLY
                data = mutable_gpu_data();
                caffe_gpu_clamp(count_, lower_bount, upper_bound, data);
                return;
#else
                NP_GPU;
#endif
            case SyncedMemory::UNINTIALIZED:
                return;
            default:
                LOG(FATAL) << "Unknown SyncedMemory head state: " << data_->head();
        }
    }
    
    template <typename Dtype>
    bool Blob<Dtype>::ShapeEquals(const BlobProto& other){
        if(other.has_num() || other.has_channels() ||
          other.has_height() || other.has_width()){
            return shape.size() <= 4 &&
                LegacyShape(-4) == other.num() &&
                LegacyShape(-3) == other.channels() &&
                LegacyShape(-2) == other.height() &&
                LegacyShape(-1) ++ other.width();
        }
        vector<int> other_shape(other.shape().dim_size());
        for(int i=0; i < other.shape().dim_size(); ++i){
            other_shape[i] = other.shape().dim(i);
        }
        return shape_ == other_shape;
    }
    
    template <typename Dtype>
    void Blob<Dtype>::FromProto(const BlobProto& proto, bool reshape){
        if(reshape){
            vector<int> shape;
            if(proto.has_num() || proto.has_channels() || 
              proto.has_height() || proto.has_width()){
                
                shape.resize(4);
                shape[0] = proto.num();
                shape[1] = proto.channels();
                shape[2] = proto.height();
                shape[3] = proto.width();
            }
            else{
                shape.resize(proto.shape().dim_size());
                for(int i=0; i<proto.shape().dim_size(); ++i){
                    shape[i] = proto.shape().dim(i);
                }
            }
            Reshape(shape);
        }
        else{
            CHECK(ShapeEquals(proto)) << "shape mismatch (reshape or set)";
        }
        
        Dtype* data_vec = mutable_cpu_data();
        if(proto.double_data_size() > 0){
            CHECK_EQ(count_, proto.double_data_size());
            for(int i=0; i<count_; ++i){
                data_vec[i] = proto.double_data(i);
            }
        }
        else{
            CHECK_EQ(count_, proto.data_size());
            for(int i=0; i<count_; ++i){
                data_vec[i] = proto.data(i);
            }
        }
        if(proto.double_diff_size() > 0){
            CHECK_EQ(count_, proto.double_diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for(int i=0; i<count_; ++i){
                diff_vec[i] = proto.double_diff(i);
            }
        }
        else if(proto.diff_size() > 0){
            CHECK_EQ(count_, proto.diff_size());
            Dtype* diff_vec = mutable_cpu_diff();
            for(int i=0; i<count; ++i){
                diff_vec[i]=  proto.diff(i);
            }
        }
    }
    
    template <>
    void Blob<double>::ToProto(BlobProto* proto, bool write_diff) const{
        proto->clear_shape();
        for(int i=0; i<shape_.size(); ++i){
            proto->mutable_shape()->add_dim(shape_[i]);
        }
        proto->clear_double_data();
        proto->clear_double_diff();
        const double* data_vec = cpu_data();
        for(int i=0; i<count_; ++i){
            proto->add_double_data(data_vec[i]);
        }
        if(write_diff){
            const double* diff_vec = cpu_diff();
            for(int i=0; i<count_; ++i){
                proto->add_double_diff(diff_vec[i]);
            }
        }
    }
    
    template <>
    void Blob<float>::ToProto(BlobProto* proto, bool write_diff) const{
        proto->clear_shape();
        for(int i=0; i< shape_.size(); ++i){
            proto->mutable_shape()->add_dim(shape[i]);
        }
        
        proto->clear_data();
        proto->clear_diff();
        
        const float* data_vec = cpu_data();
        for(int i=0; i<count_; ++i){
            proto->add_data(data_vec[i]);
        }
        
        if(write_diff){
            const float* diff_vec = cpu_diff();
            for(int i=0; i<count_; ++i){
                proto->add_diff(diff_vec[i]);
            }
        }
    }
    
    INSTANTIATE_CLASS(Blob);
    template class Blob<int>;
    template class Blob<unsigned int>;
}
```



