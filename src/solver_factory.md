# Solver factory

**solver_factory.hpp**

```c++
#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
    
    template <typename Dtype>
    class Solver;
    
    template <typename Dtype>
    class SolverRegistry{
    public:
        
        // Creator是一个函数指针类型，指向的函数的参数为SolverParameter类型，返回类型为Solver<Dtype>* 。
		// 这个函数就是 ”特定Solver对应的Creator函数"。调用这个函数可以就得到实例化的子类。 
		//（注意：容器的类型是被定义为CreatorRegistry）
        typedef Solver<Dtype>* (*Creator)(const SolverParameter&);
        typedef std::map<string, Creator> CreatorRegistry; // 注册器，key是solver的type，value是函数指针creator
        
        // Registry的作用是，调用时返回一个指向 容器（CreatorRegistry）类型的静态变量 g_registry_。
		// 说白了g_registry_就是一个容器指针。因为这个变量是static的，所以即使多次调用这个函数，也只会得到一个g_registry_，
		// 而且在其他地方修改这个map里的内容，是存储在这个map中的
        static CreatorRegistry& Registry(){
            static CreatorRegistry* g_registry_ = new CreatorRegistry(); // 只创建一次，之后存储在静态区
            return *g_registry_;
        }
        
        // Adds a creator.
		// 先调用Registry()得到容器，确保容器内不存在要添加的type关键字，然后添加Key为type，Value为Creator函数的元素。
        static void AddCreator(const string& type, Creator& creator){
            CreatorRegistry& registry = Registry();
            CHECK_EQ(registry.count(type), 0)
                << "Solver type " << type << " already registered.";
            registry[type] = creator;
        }
        
 	 	// Get a solver using a SolverParameter.
        static Solver<Dtype>* CreateSolver(const SolverParameter& param){
            const string& type = param.type();
            CreatorRegistry& registry = Registry();
            CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
                << " (known types: " << SolverTypeListString() << ")";
            return registry[type](param); // 根据type找到注册器g_registry_中的
        }
        
        static vector<string> SolverTypeList(){
            CreatorRegistry& registry = Registry();
            vector<string> solver_types;
            for(typename CreatorRegistry::iterator iter = registry.begin();
               iter != registry.end(); ++iter){
                solver_types.push_back(iter->first);
            }
            return solver_types;
        }
        
    private:
        SolverRegistry() {}
        
        static string SolverTypeListString(){
            vector<string> solver_types = SolverTypeList();
            string solver_types_str;
            for(vector<string>::iterator iter = solver_types.begin();
               iter != solver_types.end(); ++iter){
                if(iter != solver_types.begin()){
                    solver_types_str += ", ";
                }
                solver_types_str += *iter;
            }
            return solver_types_str;
        }
    };
    
    template <typename Dtype>
    class SolverRegisterer{
    public:
        SolverRegisterer(const string& type,
                        Solver<Dtype>* (*creator)(const SolverParameter&)){
            SolverRegistry<Dtype>::AddCreator(type, creator);
        }
    };
    
#define REGISTER_SOLVER_CREATOR(type, creator) \
	static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>); \
	static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>) \

#define REGISTER_SOLVER_CLASS(type) \
	template <typename Dtype>
    Solver<Dtype>* Creator_##type##Solver(const SolverParameter& param)
    {
    	return new type##Solver<Dtype>(param);    
    }
    
    REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)
}
```



