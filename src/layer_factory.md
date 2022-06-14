# Layer Factory
layer factory 在layer_factory.hpp和layer_factory.cpp中实现
**layer_factory.hpp**

```c++
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class Layer;
    
    template <typename Dtype>
    class LayerRegistry{
    public:
        
        // CreatorRegistry为一个map类型，其里面的key-value对应的值分别为string, Creator, 
        // 其中string为Layer的type，而Creator可以看到是一个函数指针，
		// 其入参为LayerParameter，其意义为相对应用的Layer creator函数用于产生相对应的Layer,
        // 该设计模式为C++的工厂模式，是caffe中用到的比较关键的模式．
		// 其中映射表主要包含三部分：
		// 1. 定义函数指针Creator，输入参数是const LayerParameter&， 
        //    返回的是Layer<Dtype>类型的指针，属于映射表的Value定义
        typedef shared_ptr<Layer<Dtype>>(*Creator)(const LayerParameter&);
        // 2. CreatorRegistry是神经网络层的名字与对应的创建该层指针的映射，映射表定义
        typedef std::map<string, Creator> CreatorRegistry;
        
        // 3. 返回映射表
		// 函数里面定义了静态变量g_registry_，由static关键字作用可知，
        // 其静态变量的存储区域是在静态变量区域，不会随着函数调用完毕后而消失，
		// 故记录Layer注册信息是存储在静态变量区中的g_registry_中，
        // 该函数仅会在第一次调用时会new申请一个CreatorRegistry空间，
		// 后面多次调用不再申请新的空间．因为g_registry_为静态变量．
        static CreatorRegistry& Registry(){
            static CreatorRegistry* g_registry_ = new CreatorRegistry();
            return *g_registry_;
        }
        
        // Adds a creator.
		// AddCreator()函数为LayerRegistry类中提供的注册layer接口,
        // 将其相对应的Layer Creator函数指针以及Layer type注册到g_registry_中．
		// 首先调用Registry()获取到g_registry_指针（仅在第一次申请新的内存，其他相当与获取到g_registry_指针）
		// 映射表添加：给定层名称以及层指针，将其加入到注册表中
        static void AddCreator(const string& type, Creator creator){
            CreatorRegistry& registry = Registry();
            CHECK_EQ(registry.count(type), 0)
                << "Layer type " << type << " already registered.";
            registry[type] = creator;
        }
        
        // Get a layer using a LayerParameter.
		// 基于映射表创建神经网络层实例：
		// 通过LayerParameter中的type(存储了层名称)，并基于param的参数返回特定层的实例智能指针
        static shared_ptr<Layer<Dtype>> CreateLayer(const LayerParameter& param){
            if(Caffe::root_solver()){
                LOG(INFO) << "Creating layer " << param.name();
            }
            const string& type = param.type();
            CreatorRegistry& registry = Registry();
            CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
                << " (known types: " << LayerTypeListString() << ")";
            return registry[type](param);
        }
        
        static vector<string> LayerTypeList(){
            CreatorRegistry& registry = Registry();
            vector<string> layer_types;
            for(typename CreatorRegistry::iterator iter = registry.begin();
               iter != registry.end(); ++iter){
                layer_types.push_back(iter->first);
            }
            return layer_types;
        }
        
    private:
        // Layer registry should never be instantiated - everything is done with its static variables.
		// 构造函数声明为private，禁止实例化，所有操作通过static变量完成
        LayerRegistry() {}
        
        static string LayerTypeListString(){
            vector<string> layer_types = LayerTypeList();
            string layer_types_str;
            for(vector<string>::iterator iter = layer_types.begin();
               iter != layer_types.end(); ++iter){
                if(iter != layer_types.begin()){
                    layer_types_str += ", ";
                }
                layer_types_str += *iter;
            }
            return layer_types_str;
        }
    };
    
    // 类LayerRegister主要是调用了类LayerRegistry的映射表添加函数，更新映射表
    template <typename Dtype>
    class LayerRegisterer{
    public:
        LayerRegisterer(const string& type,
                       shared_ptr<Layer<Dtype>>(*creator)(const LayerParameter&)){
            LayerRegistry<Dtype>::AddCreator(type, creator);
        }
    };

    // 宏定义里面: #是将输入字符串化， ##作为连接符，将前后string连接起来
#define REGISTER_LAYER_CREATOR(type, creator) \
	static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>); \
	static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>) \
	
	// REGISTER_LAYER_CLASS(Data)进行展开为如下代码：
	// template <typename Dtype>                                                
	// shared_ptr<Layer<Dtype> > Creator_DataLayer(const LayerParameter& param) 
	// {                                                                            
		 //return shared_ptr<Layer<Dtype> >(new typeData<Dtype>(param));           
	// }                                                                           
	// static LayerRegisterer<float> g_creator_f_Data(Data, Creator_DataLayer);     
	// static LayerRegisterer<double> g_creator_d_Data(Data, Creator_DataLayer);    

	// 那么问题来了既然REGISTER_LAYER_CLASS（）能够实现Layer注册功能，
    // 何时才能能够真正调用注册．还要继续补充static关键字作用，
	// static关键子修饰变量是存储在静态存储区中，不依赖于类的具体实例，它是在main函数运行之前就已经开始申请内存空间，
	// 那么既然注册最终是通过g_creator_f_Data和g_creator_d_Data,两个静态变量实现的，
    // 所以在main函数运行之前就会新建LayerRegisterer空间，
	// 进而调用到LayerRegisterer的构造函数．那么就可以得出使用REGISTER_LAYER_CLASS宏注册的类，
    // 其在main函数之前就实现了注册Layer功能，这就是LayerRegisterer设计精妙之处．
#define REGISTER_LAYER_CLASS(type) \
	template <typename Dtype> \
    shared_ptr<Layer<Dtype>> Creator_##type##Layer(const LayerParameter& param) \
    {																			\
    	return shared_ptr<Layer<Dtype>>(new type##Layer<Dtype>(param));			\
	}																			\		
	REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)
}
```

**layer_factory.cpp**

```c++
#ifdef WITH_PYTHON_LAYER
#include <boost/python.hpp>
#endif
#include <string>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/bn_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#ifdef USE_CUDNN
#include "caffe/layers/cudnn_bn_layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"


#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#endif

#ifdef WITH_PYTHON_LAYER
#include "caffe/layers/python_layer.hpp"
#endif

namespace caffe{
	template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetConvolutionLayer(const LayerParameter& param){
        ConvolutionParameter conv_param = param.convolution_param();
        ConvolutionParameter_Engine engine = conv_param.engine();
#ifdef USE_CUDNN
        bool use_dilation = false;
        for(int i=0; i<conv_param.dilation_size(); ++i){
        	if(conv_param.dilation(i) > 1){
                use_dialtion = true;
            }
        }
#endif
        if(engine == ConvolutionParameter_Engine_DEFAULT){
            engine = ConvolutionParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            if(!use_dilation){
                engine = CpnvolutionParameter_Engine_CUDNN;
            }
#endif
        }
        if(engine == ConvolutionParamter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new ConvolutionLayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == ConvolutionParameter_Engine_CUDNN){
            if(use_dilation){
                LOG(FATAL) << "CuDNN doesn't support the dilated convolution at Layer "
                    << param.name();
            }
            return shared_ptr<Layer<Dtype>>(new CuDNNConvolutionLayer<Dtype>(param));
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << "has unknown engine.";
            throw;
        }
    }
    
    REGISTER_LAYER_CREATOR(Convolution, GetConvolutionLayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetBNLayer(const LayerParameter& param){
        BNParameter_Engine engine = param.bn_param().engine();
        if(engine == BNParameter_Engine_DEFAULT){
            engine = BNParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = BNParameter_Engine_CUDNN;
#endif
        }
        if(engine == BNParameter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new BNLayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == BNParameter_Engine_CUDNN){
            return shared_ptr<Layer<Dtype>>(new CuDNNBNLayer<Dtype>(param));
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unkown engine.";
        }
    }
    
    REGISTER_LAYER_CREATOR(BN, GetBNLayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetPoolingLayer(const LayerParameter& param){
        PoolingParameter_Engine engine = param.pooling_param().engine();
        if(engine == PoolingParameter_Engine_DEFAULT){
            engine = PoolingParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = PoolingParameter_Engine_CUDNN;
#endif
        }
        if(engine == PoolingParameter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == PoolingParameter_Engine_CUDNN){
            if(param.top_size() > 1){
                LOG(INFO) << "cuDNN does not support multiple tops. "
                    << "Using Caffe's own pooling layer.";
                return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
            }
            
            if(param.pooling_param().pool() == PoolingParameter_PoolMethod_MAX){
                return shared_ptr<Layer<Dtype>>(new PoolingLayer<Dtype>(param));
            }
            else{
                return shared_ptr<Layer<Dtype>>(new CuDNNPoolingLayer<Dtype>(param));
            }
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
            throw;
        }
    }
    
    REGISTER_LAYER_CREATOR(Pooling, GetPoolingLayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetLRNLayer(const LayerParameter& param){
        LRNParameter_Engine engine = param.lrn_param().engine();
        
        if(engine == LRNParameter_Engine_DEFAULT){
#ifdef USE_CUDNN
            engine = LRNParameter_Engine_CUDNN;
#else
            engine = LRNParameter_Engine_CAFFE;
#endif
        }
        if(engine == LRNParameter_Engine_CUDNN){
            LRNParameter lrn_param = param.lrn_param();
            
            if(lrn_param.norm_region() == LRNParameter_NormRegion_WITHIN_CHANNEL){
                return shared_ptr<Layer<Dtype>>(new CuDNNLCNLayer<Dtype>(param));
            }
            else{
                if(param.lrn_param().local_size() > CUDNN_LRN_MAX_N){
                    return shared_ptr<Layer<Dtype>>(new LRNLayer<Dtype>(param));
                }
                else{
                    return shared_ptr<Layer<Dtype>>(new CuDNNLRNLayer<Dtype>(param));
                }
            }
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
            throw;
        }
    }
    REGISTER_LAYER_CREATOR(LRN, GetLRNLayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetReLULayer(const LayerParameter& param){
        ReLUParameter_Engine engine = param.relu_param().engine();
        if(engine == ReLUParameter_Engine_DEFAULT){
            engine = ReLUParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = ReLUParameter_Engine_CUDNN;
#endif
        }
        if(engine == ReLUParameter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new ReLULayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == ReLUParameter_Engine_CUDNN){
            return shared_ptr<Layer<Dtype>>(new CuDNNReLULayer<Dtype>(param));
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
            throw;
        }
    }
    REGISTER_LAYER_CREATOR(ReLU, GetReLULayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetSigmoidLayer(const LayerParameter& param){
        SigmoidParameter_Engine engine = param.sigmoid_param().engine();
        if(engine == SigmoidParameter_Engine_DEFAULT){
            engine = SigmoidParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = SigmoidParameter_Engine_CUDNN;
#endif
        }
        if(engine == SigmoidParameter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new SigmoidLayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == SigmoidParameter_Engine_CUDNN){
            return shared_ptr<Layer<Dtype>>(new CuDNNSigmoidLayer<Dtype>(param));
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
            throw;
        }
    }
    REGISTER_LAYER_CREATOR(Sigmoid, GetSigmoidLayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetSoftmaxLayer(const LayerParameter& param){
        SoftmaxParameter_Engine engine = param.softmax_param().engine();
        if(engine = SoftmaxParameter_Engine_DEFAULT){
            engine = SoftmaxParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = SoftmaxParameter_Engine_CUDNN;
#endif
        }
        if(engine == SoftmaxParameter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new SoftmaxLayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == SoftmaxParameter_Engine_CUDNN){
            return shared_ptr<Layer<Dtype>>(new CuDNNSoftLayer<Dtype>(param));
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
            throw;
        }
    }
    REGISTER_LAYER_CREATOR(Softmax, GetSoftmaxLayer);
    
    template <typename Dtype>
    shared_ptr<Layer<Dtype>> GetTanHLayer(const LayerParameter& param){
        TaHNParameter_Engine engine = param.tanh_param().engine();
        if(engine == TanHParameter_Engine_DEFAULT){
            engine = TanHParameter_Engine_CAFFE;
#ifdef USE_CUDNN
            engine = TanHParameter_Engine_CUDNN;
#endif
        }
        if(engine == TanHParameter_Engine_CAFFE){
            return shared_ptr<Layer<Dtype>>(new TanHLayer<Dtype>(param));
#ifdef USE_CUDNN
        }
        else if(engine == TanHParameter_Engine_CUDNN){
            return shared_ptr<Layer<Dtype>>(new CuDNNTanHLayer<Dtype>(param));
#endif
        }
        else{
            LOG(FATAL) << "Layer " << param.name() << " has unknown engine.";
            throw;
        }
    }
    
    REGISTER_LAYER_CREATOR(TanH, GetTanHLayer);
    
}
```



