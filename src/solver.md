# Solver

**solver.hpp**

```c++
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe{
    
    // 枚举了一些动作，有时候我们的客户端可能会提前终止程序，
    // 例如我们使用 ctrl+c把程序终止了，他就需要做出一些响应，例如生成快照，快速保存等等。
    namespace SolverAction{
        enum Enum{
            NONE = 0,  // None就是没有什么异常
            STOP = 1,  // STOP就是训练停止了
            SNAPSHOT = 2 // SNAPSHOT就是创建一个SNAPSHOT然后我们接着去训练
        };
    }
    
    // Type of a function that returns a Solver Action enumeration.
    typedef boost::function<SolverAction::Enum()> ActionCallback;
    
    template <typename Dtype>
    class Solver{
   	public:
        // 显示构造函数, 内部会调用Init函数
        explicit Solver(const SolverParameter& param);
        explicit Solver(const string& param_file);
        
        // 成员变量赋值，包括param_、iter_、current_step_,并调用InitTrainNet和InitTestNets函数
        void Init(const SolverParameter& param);
        
        // 为成员变量net_赋值
        void InitTrainNet();
        // 为成员变量test_nets_赋值
        void InitTestNets();
        
        // Client of the Solver optionally may call this in order to set the function
		// that the solver uses to see what action it should take (e.g. snapshot or
		// exit training early).
        void SetActionFunction(ActionCallback func);
        SolverActon::Enum GetRequestedAction();
        
        // The main entry of the solver function. In default, iter will be zero. Pass
		// in a non-zero iter number to resume training for a pre-trained net.
        virtual void Solve(const char* resume_file = NULL);
        virtual void Solve(const string resume_file) {Solve(resume_file.c_str());}
        
        // Make and apply the update value for the current iteration.
		// 更新net的权值和偏置
        virtual void ApplyUpdata() = 0;
        
        // 反复执行net前向传播反向传播计算,期间会调用函数ApplyUpdate、Snapshot及类Callback两个成员函数
        void Step(int iters);
        
        // The Restore method simply dispatches to one of the
		// RestoreSolverStateFrom___ protected methods. You should implement these
		// methods to restore the state from the appropriate snapshot type.
		// 加载已有的模型
        void Restore(const char* resume_file);
        
        // The Solver::Snapshot function implements the basic snapshotting utility
		// that stores the learned net. You should implement the SnapshotSolverState()
		// function that produces a SolverState protocol buffer that needs to be
		// written to disk together with the learned net.
		// 快照，内部会调用SnapshotToBinaryProto或SnapshotToHDF5、SnapshotSolverState函数
        void Snapshot();
        
        virtual ~Solver();
        
        inline const SolverParameter& param() const {return param_;}
        inline shared_ptr<Net<Dtype>> net() {return net_;}
        
        inline const vector<shared_ptr<Net<Dtype>>>& test_nets() {
            return test_nets_;
        }
        
        int iter() const {return iter_;}
        
        class Callback{
        protected:
            virtual void on_start() = 0;
            virtual void on_gradients_ready() = 0;
            
            template <typename T>
            friend class Solver;
        };
        
        const vector<Callback*>& callbacks() const {return callbacks_;}
        
        void add_callback(Callback* value){
            callbacks_.push_back(value);
        }
        
        void CheckSnapshotWritePermissions();
        
        virtual inline const char* type() const {return "";}
        
    protected:
        string SnapshotFilename(const string extension);
        string SnapshotToBinaryProto();
        string SnapshotToHDF5();
        
        void TestAll();
        void Test(const int test_net_id = 0);
        
        virtual void SnapshotSolverState(const string& model_filename) = 0;
        virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
        virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
        void DisplayOutputBlobs(const int net_id);
        void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);
        
        SolverParameter param_;
        int iter_;
        int current_step_;
        shared_ptr<Net<Dtype>> net_;
        vector<shared_ptr<Net<Dtype>>> test_nets_;
        vector<Callback*> callbacks_;
        vector<Dtype> losses_;
        Dtype smoothed_loss_;
        
        // A function that can be set by a client of the Solver to provide indication
		// that it wants a snapshot saved and/or to exit early.
        ActionCallback action_request_function_;
        
        // True if a request to stop early was received.
        bool requested_early_exit_;
        
        Timer iteration_timer_;
        float iteration_last_;
        
        DISABLE_COPY_AND_ASSIGN(Solver);
    };
}
```

**solver.cpp**

```c++
#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe{
    
    template <typename Dtype>
    void Solver<Dtype>::SetActionFunction(ActionCallback func){
        action_request_function_ = func;
    }
    
    template <typename Dtype>
    SolverAction::Enum Solver<Dtype>::GetRequestedAction(){
        if(action_request_function_){
            return action_request_function_();
        }
        
        return SolverAction::None;
    }
    
    template <typename Dtype>
    Solver<Dtype>::Solver(const SolverParameter& param)
        : net_(), callbacks_(), requested_early_exit_(false){
            Init(param);
    }
    
    template <typename Dtype>
    Solver<Dtype>::Solver(const string& param_file)
        : net_(), callbacks_(), requested_early_exit_(false){
    	SolverParam param;
        ReadSolverParamsFromTextFileOrDie(param_file, &param);
        Init(param);
    }
    
    template <typename Dtype>
    void Solver<Dtype>::Init(const SolverParameter& param){
        LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
            << std::endl << param.DebugString();
        
        param_ = paramp;
        CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
        CheckSnapshotWritePermissions();
        if(param_.random_seed() >= 0){
            Caffe::set_random_seed(param_.random_seed() + Caffe::solver_rank());
        }
        
        InitTrainNet();
        InitTestNets();
        
        if(Caffe::root_solver()){
            LOG(INFO) << "Solver scaffolding done.";
        }
        
        iter_ = 0;
        current_step_ = 0;
    }
    
    template <typename Dtype>
    void Solver<Dtype>::InitTrainNet(){
        const int num_train_nets = param_.has_net() + param_.has_net_param() + 
            param_.has_train_net() + param_.has_train_net_param();
        
        const string& field_names = "net, net_param, train_net, train_net_param";
        CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
            << "using one of these fields: " << field_names;
        CHECK_LE(num_train_nets, 1) << "SolverParamter must not contain more than "
            << "one of these fields specifying a train_net: " << field_names;
        
        NetParameter net_param;
        if(param_.has_train_net_param()){
            LOG_IF(INFO, Caffe::root_solver())
                << "Creating training net specified in train_net_param.";
            net_param.CopyFrom(param_.train_net_param());
        }
        else if(param_.has_net_param()){
            LOG_IF(INFO, Caffe::root_solver())
                << "Creating training net specified in net_param.";
            net_param.CopyFrom(param_.net_param());
        }
        if(param.has_net()){
            LOG_IF(INFO, Caffe::root_solver())
                << "Creating training net specified in net_param.";
            net_param.CopyFrom(param_.net_param());
        }
        if(param_.has_net()){
            LOG_IF(INFO, Caffe::root_solver())
                << "Creating training net from net file: " << param_.net();
            ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
        }
        
        // Set the correct NetState.  We start with the solver defaults (lowest
		// precedence); then, merge in any NetState specified by the net_param itself;
		// finally, merge in any NetState specified by the train_state (highest
		// precedence).
        NetState net_state;
        net_state.set_phase(TRAIN);
        net_state.MergeFrom(net_param.state());
        net_state.MergeFrom(param_.train_state());
        net_param.mutable_state()->CopyFrom(net_state);
        net_.reset(new Net<Dtype>(net_param));
    }
    
    template <typename Dtype>
    void Solver<Dtype>::InitTestNets(){
        const bool has_net_param = param_.has_net_param();
        const bool has_net_file = param_.has_net();
        const int num_generic_nets = has_net_param + has_net_file;
        CHECK_LE(num_generic_nets, 1) 
            << "Both net_param and net_file may not be specified.";
        const int num_test_net_params = param_.test_net_param_size();
        const int num_test_net_files = param_.test_net_size();
        const int num_test_nets = num_test_net_params + num_test_net_files;
        if(num_generic_nets){
            CHECK_GE(param_.test_iter_size(), num_test_nets)
                << "test_iter must be specified for each test network.";
        }
        else{
            CHECK_EQ(param_.test_iter_size(), num_test_nets)
                << "test_iter must be specified for each test network."
        }
        
        const int num_generic_net_instances = param_.test_iter_size() _ num_test_nets;
        const int num_test_net_instances = num_test_nets + num_generic_net_instances;
        
        if(param_.test_state_size()){
            CHECK_EQ(param_.test_state_size(), num_test_net_instances)
                << "test_state must be unspecified or specified once per test net.";
        }
        
        if(num_test_net_instances){
            CHECK_GT(param_.test_interval(), 0);
        }
        
        int test_net_id = 0;
        vector<string> sources(num_test_net_instances);
        for(int i=0; i < num_test_net_params; ++i, ++ test_net_id){
            sources[test_net_id] = "test_net_param";
            net_params[test_net_id].CopyFrom(param_.test_net_param(i));
        }
        for(int i=0; i<num_test_files; ++i, ++test_net_id){
            sources[test_net_id] = "test_net_file: " + param.test_net(i);
            ReadNetParamsFromTextFileOrDie(param_.test_net(i), &net_params[test_net_id]);
        }
        const int remaining_test_nets = param_.test_iter_size() - test_net_id;
		if(has_net_param){
            for(int i=0; i < remaining_test_nets; ++i, ++test_net_id){
                sources[test_net_id] = "net_param";
                net_params[test_net_id].CopyFrom(param_.net_param());
            }
        }        
        if(has_net_file){
            for(int i=0; i < remaining_test_nets; ++i, ++test_net_id){
                sources[test_net_id] = "net file: " + param_.net();
                ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
            }
        }
        
        test_nets_.resize(num_test_net_instances);
        for(int i=0; i<num_test_net_instances; ++i){
            NetState net_state;
            net_state.set_phase(TEST);
            net_state.MergeFrom(net_patams[i].state());
            
            if(param_.test_state_size()){
                net_state.MergeFrom(param_.test_state(i));
            }
            
            net_params[i].mutable_state()->CopyFrom(net_state);
            LOG(INFO)
                << "Creating test net (#" << i << ") specified by " << sources[i];
            test_nets_[i].reset(new Net<Dtype>(net_params[i]));
            test_nets_[i]->set_debug_info(param_.debug_info());
        }
    }
    
    template <typename Dtype>
    void Solver<Dtype>::Step(int iters){
        // 设置开始的迭代次数(如果是从之前的snapshot恢复的，
		// 那iter_等于snapshot时的迭代次数)和结束的迭代次数
        const int start_iter = iter_;
        const int stop_iter = iter_ + iters;
        
        // 输出的loss为前average_loss次loss的平均值，在solver.prototxt里设置，默认为1，
		// losses存储之前的average_loss个loss，smoothed_loss为最后要输出的均值
        int average_loss = this->param_.average_loss();
        losses_.clear();
        smoothed_loss_ = 0;
        iteration_timer_.Start();
        
        // 迭代
        while(iter_ < stop_iter){
            // zero-init the params
			// 清空上一次所有参数的梯度
            net_->ClearParamDiffs();
            
            // 判断是否需要测试
            if(param_.test_interval() && iter_%param_.test_interval() == 0
              && (iter_ > 0 || param_.test_initialization())){
                if(Caffe::root_solver()){
					TestAll();
                }
                if(requested_early_exit_){
                    // Break out of the while loop because stop was requested while testing.
                    break;
                }
            }
            
            for(int i=0; i < callbacks_.size(); ++i){
                callbacks_[i]->on_start();
            }
            
            // 判断当前迭代次数是否需要显示loss等信息
            const bool display = param_.display() && iter % param_.display() == 0;
            net_->set_debug_info(display && param_.debug_info());
            
            // accumulate the loss and gradient
			// iter_size也是在solver.prototxt里设置，实际上的batch_size=iter_size*网络定义里的batch_size，
			// 因此每一次迭代的loss是iter_size次迭代的和，再除以iter_size，
            // 这个loss是通过调用`Net::ForwardBackward`函数得到的
			// 这个设置我的理解是在GPU的显存不够的时候使用，比如我本来想把batch_size设置为128，但是会out_of_memory，
			// 借助这个方法，可以设置batch_size=32，iter_size=4，那实际上每次迭代还是处理了128个数据
            Dtype loss = 0;
            for(int i=0; i<param_.iter_size(); ++i){
                // 调用了Net中的代码，主要完成了前向后向的计算，
				// 前向用于计算模型的最终输出和Loss，后向用于
				// 计算每一层网络和参数的梯度。
                loss += net_->ForwardBackward(); // 这行代码通过Net类的net_指针调用其成员函数ForwardBackward()
            }
            
            loss /= param_.iter_size();
            if(isnan(loss)){
                LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_ 
                    << ", loss = " << loss << ", ignore and continue";
                ++iter_;
                continue;
			}
            
            
			// 这个函数主要做Loss的平滑。由于Caffe的训练方式是SGD，我们无法把所有的数据同时
			// 放入模型进行训练，那么部分数据产生的Loss就可能会和全样本的平均Loss不同，在必要
			// 时候将Loss和历史过程中更新的Loss求平均就可以减少Loss的震荡问题。
            UpdateSmoothedLoss(loss, start_iter, average_loss);
            
            // 输出当前迭代的信息
            if(display){
                float lapse = iteration_timer_.Seconds();
                float per_s = (iter_ - iteration_last_) / (lapse ? lapse : 1);
                LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
                    << " (" << per_s << " iters/s, " << lapse << "s/"
                    << param_.display() << " iters), loss = " << smoothed_loss_;
                iteration_timer_.Start();
                iteration_last_ = iter_;
                const vector<Blob<Dtype>*>& result = net_->output_blobs();
                int score_index = 0;
                for(int j=0; j<result.size(); ++j){
                    const Dtype* result_vec = result[j]->cpu_data();
                    const string& output_name = 
                        net_->blob_names()[net_->output_blob_indices()[j]];
                    const Dtype loss_weight = net_->blob_loss_weightd()[net_->output_blob_indices()[j]];
                    
                    for(int k=0; k<result[j]->count(); ++k){
                        ostringstream loss_msg_stream;
                        if(loss_weight){
                            loss_msg_stream << " (* " << loss_weight 
                                << " = " << loss_weight * result_vec[k] << " loss)";
                        }
                        LOG_IF(INFO, Caffe::root_solver()) << " Train net output #"
                            << score_index++ << " : " << output_name << " = "
                            << result_vec[k] << loss_msg_stream.str());
                    }
                }
            }
            
            for(int i=0; i<callbacks_.size(); ++i){
				callbacks_[i]->on_gradients_ready();
            }
            
            // 执行梯度的更新，这个函数在基类`Solver`中没有实现，会调用每个子类自己的实现。
            ApplyUpdate();
            ++iter_;
			
            SolverAction::Enum request = GetRequestedAction();
            
            // Save a snapshot if needed.
            if((param_.snapshot()
              && iter_ % param_.snapshot() == 0
              && Caffe::root_solver()) ||
              (request == SolverAction::SNAPSHOT)){
                Snapshot();
            }
            if(SolverAction::STOP == request){
                requested_early_exit_ = true;
                break;
            }
        }
    }
    
    // 对整个网络进行训练（也就是你运行Caffe训练某个模型）的时候，实际上是在运行caffe.cpp中的train()函数，
	// 而这个函数实际上是实例化一个Solver对象，初始化后调用了Solver中的Solve()方法  
	// 调用此方法训练网络，其中会调用Step()方法来迭代，迭代 param_.max_iter() - iter_ 次
    template <typename Dtype>
    void Solver<Dtype>::Solve(const char* resume_file){
        CHECK(Caffe::root_solver());
        LOG(INFO) << "Solving " << net_->name();
        LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();
        
        // Initialize to false every time we start solving.
        requested_early_exit_ = false;
        
        // 判断`resume_file`这个指针是否NULL，如果不是则需要从resume_file存储的路径里读取之前训练的状态
        if(resume_file){
            LOG(INFO) << "Restoring previous solver status from " << resume_file;
            Restore(resume_file);
        }
        
        int start_iter = iter_;
        Step(param_.max_iter() - iter_);
        
        if(param_.snapshot_after_train() 
           && (!param_.snapshot() || iter_ % param_.snapshot() != 0)){
            Snapshot();
        }
        
        if(requested_early_exit_){
            LOG(INFO) << "Optimization stopped early.";
            return;
        }
        
        // After the optimization is done, run an additional train and test pass to
		// display the train and test loss/outputs if appropriate (based on the
		// display and test_interval settings, respectively).  Unlike in the rest of
		// training, for the train net we only run a forward pass as we've already
		// updated the parameters "max_iter" times -- this final pass is only done to
		// display the loss, which is computed in the forward pass.
        if(param_.display() && iter_ % param_.display() == 0){
			int average_loss = this->param_.average_loss();
            Dtype loss;
            net_->Forward(&loss);
            
            UpdateSmoothedLoss(loss, start_iter, average_loss);
            LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
        }
        
        if(param_.test_interval() && iter_ % param_.test_interval() == 0){
            TestAll();
        }
        LOG(INFO) << "Optimization Done.";
    }
    
    template <typename Dtype>
    void Solver<Dtype>::TestAll(){
        for(int test_net_id = 0;
           test_net_id < test_nets_.size() && !requested_early_exit_; 
           ++test_net_id){
            Test(test_net_id);
        }
    }
    
    template <typename Dtype>
    void Solver<Dtype>::Test(const int test_net_id){
        CHECK(Caffe::root_solver());
        LOG(INFO) << "Iteration " << iter_ << ", Testing net (# " << test_net_id << ")";
        CHECK_NOTNULL(test_nets_[test_net_id].get())->ShareTrainedLayerWith(net_.get());
        
        vector<Dtype> test_score;
        vector<int> test_score_output_id;
        const shared_ptr<Net<Dtype>>& test_net = test_nets[test_net_id];
        Dtype loss = 0;
        for(int i=0; i<param_.test_iter(test_net_id); ++i){
            SolverAction::Enum request = GetRequestedAction();
            while(request != SolverAction::NONE){
                if(SolverAction::SNAPSHOT == request){
                    Snapshot();
                }
                else if(SolverAction::STOP == request){
                    requested_early_exit_ = true;
                }
                request = GetRequestedAction();
            }
            if(requested_early_exit_){
                break;
            }
            
            Dtype iter_loss;
            const vector<Blob<Dtype>*>& result = test_net->Forward(&iter_loss);
            if(param_.test_compute_loss()){
                loss += iter_loss;
            }
            if(i==0){
                for(int j=0; j<result.size(); ++j){
                    const Dtype* result_vec = result[j]->cpu_data();
                    for(int k=0; k<result[j]->count(); ++k){
                        test_score.push_back(result_vec[k]);
                        test_score_output_id.push_back(j);
                    }
                }
            }
            else{
                int idx = 0;
                for(int j=0; j<result.size(); ++j){
                    const Dtype* result_vec = result[j]->cpu_data();
                    for(int k=0; k<result[j]->count(); ++k){
                        test_score[idx++] += result_vec[k];
                    }
                }
            }
        }
        if(requested_early_exit_){
            LOG(INFO) << "Test interrupted.";
            return;
        }
        if(param_.test_compute_loss()){
            loss /= param_.test_iter(test_net_id);
            LOG(INFO) << "Test loss: " << loss;
        }
        for(int i=0; i < test_score.size(); ++i){
            const int output_blob_index = 
                test_net->output_blob_indices()[test_score_output_id[i]];
            const string& output_name = test_net->blob_names()[output_blob_index];
            const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
            ostringstream loss_msg_stream;
            const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
            if(loss_weight){
                loss_msg_stream << " (* " << loss_weight 
                    << " = " << loss_weight * mean_score << " loss)";
            }
            LOG(INFO) << " Test net output #" << i << ": " << output_name << " = "
                << mean_score << loss_msg_stream.str();
        }
    }
    
    template <typename Dtype>
    void Solver<Dtype>::Snapshot(){
        CHECK(Caffe::root_solver());
        string model_filename;
        switch(param_.snapshot_format()){
            case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
                model_filename = SnapshotToBinaryProto();
                break;
            case caffe::SolverParameter_SnapshotFormat_HDF5:
                model_filenam = SnapshotToHDF5();
                break;
            default:
                LOG(FATAL) << "Unsupported snapshot format.";
        }
        SnapshotSolverState(model_filename);
    }
    
    template <typename Dtype>
    void Solver<Dtype>::CheckSnapshotWritePermissions(){
        if(Caffe::root_solver() && param_.snapshot()){
            CHECK(param_.has_snapshot_prefix())
                << "In solver params, snapshot is specified but snapshot_prefix is not";
            string probe_filename = SnapshotFilename("./tempfile");
            std::ofstream probe_ofs(probe_filename.c_str());
            if(probe_ofs.good()){
                probe_ofs.close();
                std::remove(probe_filename.c_str());
            }
            else{
                LOG(FATAL) << "Cannot write to snapshot prefix '"
                    << param_.snapshot_prefix() << "'. Make sure "
                    << "that the dictionary exists and is writeable.";
            }
        }
    }
    
    template <typename Dtype>
    string Solver<Dtype>::SnapshotFilename(const string extension){
        return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_) + extension;
    }
    
    template <typename Dtype>
    string Solver<Dtype>::SnapshotToBinaryProto(){
        string model_filename = SnapshotFilename(".caffemodel");
        LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
        NetParameter net_param;
        net_->ToProto(&net_param, param_.snapshot_diff());
        WriteProtoToBinaryFile(net_param, model_filename);
        return model_filename;
    }
    
    template <typename Dtype>
    string Solver<Dtype>::SnapshotToHDF5(){
        string model_filename = SnapshotFilename(".caffemodel.h5");
        LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
        net_->ToHDF5(model_filename, param_.snapshot_diff());
        return model_filename;
    }
    
    template <typename Dtype>
    void Solver<Dtype>::Restore(const char* state_file){
        string state_filename(state_file);
        if(state_filename.size() >= 3 &&
          state_filename.compare(state_filename.size()-3, 3, ".h5") == 0){
            RestoreSolverStateFromHDF5(state_filename);
        }
        else{
            RestoreSolverStateFromBinaryProto(state_filename);
        }
    }
    
    template <typename Dtype>
    void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss){
        if(losses_.size() < average_loss){
            losses_.push_back(loss);
            int size = losses_.size();
            smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
        }
        else{
            int idx = (iter - start_iter) % average_loss;
            smoothed_loss_ += (loss - losses_[idx]) / average_loss;
            losses_[idx] = loss;
        }
    }
    
    INSTANTIATE_CLASS(Solver);
}
```

