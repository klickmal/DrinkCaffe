# Memory Data Layer
Memory data layer在memory_data_layer.hpp 和 memory_data_layer.cpp中实现

**memory_data_layer.hpp**

```c++
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

namespace caffe{
    
    template <typename Dtype>
    class MemoryDataLayer: public BaseDataLayer<Dtype>{
    public:
        explicit MemoryDataLayer(const LayerParameter& param)
            : BaseDataLayer<Dtype>(param), has_new_data_(false){}
        
        virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const {return "MemoryData";}
        virtual inline int ExactNumBottomBlobs() const {return 0;}
        virtual inline int ExactNumTopBlobs() const {return 2;}
        
        virtual void AddDatumVector(const vector<Datum>& datum_vector);
 
#ifdef USE_OPENCV
        virtual void AddMatVector(const vector<cv::Mat>& mat_vactor,
                                  const vector<int>& labels);
#endif
        
        void Reset(Dtype* data, Dtype* label, int n);
        void set_batch_size(int new_size);
        void set_spatial_size(int new_height, int new_width);
        
        int batch_size() {return batch_size_};
        int channels() {return channels_;}
        int height() {return height_;}
        int width() {return width_;}
        
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);
        int batch_size_, channels_, height_, width_, size_;
        Dtype* data_;
        Dtype* labels_;
        int n_;
        size_t pos_;
        Blob<Dtype> added_data_;
        Blob<Dtype> added_label_;
        
        bool has_new_data_;
        bool transpose_;
    };
}
```

**memory_data_layer.cpp**

```c++
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV

#include <vector>

#include "caffe/layers/memory_data_layer.hpp"

namespace caffe{
    
	template <typename Dtype>
    void MemoryDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top){
        batch_size_ = this->layer_param_.memory_data_param().batch_size();
        channels_ = this->layer_param_.memory_data_param().channels();
        height_ = this->layer_param_.memory_data_param().height();
        width_ = this->layer_param_.memory_data_param().width();
        transpose_ = this->layer_param_.memory_data_param().transpose();
        size_ = channels_ * height_ * width_;
        
        CHECK_GT(batch_size_ * size_, 0) << 
            "batch_size_, channels, height, and width must be specified and"
            " positive in memory_data_param";
        
        int crop_size = this->transform_param_.crop_size();
        if(crop_size > 0){
            top[0]->Reshape(batch_size_, channels_, crop_size, crop_size);
            added_data_.Resahpe(batch_size_, channels_, crop_size, crop_size);
        }
        else{
            top[0]->Reshape(batch_size_, channels_, height_, width_);
            added_data_.Resahpe(batch_size_, channels_, height_, width_);
        }
        vector<int> label_shape(1, batch_size_);
        top[1]->Reshape(label_shape);
        added_label_.Reshape(label_shape);
        data_ = NULL;
        labels_ = NULL;
        added_data_.cpu_data();
        added_label_.cpu_data();
    }
    
    template <typename Dtype>
    void MemoryDataLayer<Dtype>::AddDatumVector(const vector<Datum>& datum_vector){
        CHECK(!has_new_data_) << 
            "Can't add data until current data has been consumed.";
        size_t num = datum_vector.size();
        channels_ = num;
        CHECK_GT(num, 0) << "There is no datum to add.";
        
        int crop_size = this->transform_param_.crop_size();
        if(crop_size > 0)
            added_data_.Reshape(num, channels_, crop_size, crop_size);
        else
            added_data_.Reshape(num, channels_, height_, width_);
        
        added_label_.Reshape(num, 1, 1, ,1);
        this->data_transformer_->Transform(datum_vector, &added_data_);
        Dtype* top_label = added_label_.mutable_cpu_data();
        
        for(int item_id = 0; item_id < num; ++item_id){
            top_label[item_id] = datum_vector[item_id].label();
        }
        Dtype* top_data = added_data_.mutable_cpu_data();
        Reset(top_data, top_label, num);
        has_new_data_ = true;
    }
    
#ifdef USE_OPENCV
    template <typename Dtype>
    void MemoryDataLayer<Dtype>::AddMatVector(const vector<cv::Mat>& mat_vector,
                                             const vector<cv::Mat>& labels){
        size_t num = mat_vector.size();
        CHECK(!ha_new_data_) <<
            "Can't add mat untill current data has beed consumed.";
        
        CKECK_GT(num, 0) << "There is no mat to add";
        batch_size_ = num;
        height_ = mat_vector[0].rows;
        width_ = mat_vector[0].cols;
        if(transpose_){
            std::swap(height_, width_);
        }
        int crop_size = this->transform_param_,crop_size();
        if(crop_size > 0)
            added_data_.Reshape(num, channels_, crop_size, crop_size);
        else{
            added_data_.Reshape(num, channels_, height_, width_);
        }
        
        added_label_.Reshape(num, 1, 1, 1);
        this->data_transformer_->Transform(mat_vector, &added_data_, transpose_);
        
        Dtype* top_label = added_label_.mutable_cpu_data();
        for(int item_id = 0; item_id < num && item_id < labels.size(); ++item_id){
            top_label[item_id] = labels[item_id];
        }
        
        Dtype* top_data = added_data_.mutable_cpu_data();
        Reset(top_data, top_label, num);
        has_new_data_ = true;
    }
    
# endif
    
    template <typename Dtype>
    void MemoryDataLayer<Dtype>::Reset(Dtype* data, Dtype* labels, int n){
        CHECK(data);
        CHECK(labels);
        if(this->layer_param_.has_transform_param()){
            
        }
        data_ = data;
        labels_ = labels;
        n_ = n;
        pose_ = 0;
    }
    
    template <typename Dtype>
    void MemoryDataLayer<Dtype>::set_batch_size(int new_size){
        CHECK(!has_new_data_) << 
            "Can't change batch size until current data has been consumed.";
        batch_size_ = new_size;
        added_data_.Reshape(batch_size_, channels_, height_, width_);
        added_label_.Reshape(batch_size_, 1, 1, 1);
    }
    
    template <typename Dtype>
    void MemoryDataLayer<Dtype>::set_spatial_size(int new_height, int new_width){
        CHECK(!has_new_data_) << 
            "Can't change batch_size until current data has been consumed";
        height_ = new_height;
        width_ = new_width;
        added_data_.Reshape(batch_size_, channels_, height_, width_);
    }
    
    template <typename Dtype>
    void MemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top){
        CHECK(data_) << "MemoryDataLayer needs to be initalized by calling Reset";
        int crop_size = this->transform_param_.crop_size();
        if(crop_size > 0)
            top[0]->Reshape(batch_size_, channels_, crop_size, crop_size);
        else
            top[0]->Reshape(batch_size_, channels_, height_, width_);
        
        top[1]->Reshape(batch_size_, 1, 1, 1);
        top[0]->set_cpu_data(data_ + pose_ * size_);
        top[1]->set_cpu_data(labels_ + pos_);
        pos_ = (pos_ + batch_size_) % n_;
        if(pose_ == 0)
            has_new_data_ = false;
    }
    
    INSTANTIATE_CLASS(MemoryDataLayer);
    REGISTER_LAYER_CLASS(MemoryData);
}

```

