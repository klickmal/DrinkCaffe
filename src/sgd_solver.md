# SGD_Solver

**sgd_solver.hpp**

```c++
#ifndef CAFFE_SGD_SOLVERS_HPP_
#define CAFFE_SGD_SOLVERS_HPP_

#include <string>
#include <vector>

#include "caffe/solver.hpp"

namespace caffe{
    
    // Optimizes the parameters of a Net using
	// stochastic gradient descent (SGD) with momentum.
    template <typename Dtype>
    class SGDSolver: public Solver<Dtype>{
    public:
        // 显示构造函数，调用PreSolve函数
        explicit SGDSolver(const SolverParameter& param)
        	:Solver<Dtype>(param){PreSolve();}
        explicit SGDSolver(const string& param_file)
            :Solver<Dtype>(param_file){PreSolve();}
        virtual inline const char* type() const {return "SGD";}
        
        // 获取history数据
        const vector<shared_ptr<Blob<Dtype>>>& history() {return history_;}
        
    protected:
        // 成员变量history_, update_, temp_初始化
        void PreSolve();
        // 获取学习率
        Dtype GetLearningRate();
        // 内部会调用ClipGradients、Normalize、Regularize、ComputeUpdateValue，更新net权值和偏置
        virtual void ApplyUpdate();
        virtual void Normalize(int param_id);
        virtual void Regularize(int param_id);
        // 计算并更新相应Blob值，调用caffe_cpu_axpby和caffe_copy函数
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        // clip parameter gradients to that L2 norm，如果梯度值过大，就会对梯度做一个修剪，
		// 对所有的参数乘以一个缩放因子，使得所有参数的平方和不超过参数中设定的梯度总值
        virtual void ClipGradients();
        virtual void SnapshotSolverState(const string& model_filename);
        virtual void SnapshotSolverStateToBinaryProto(const string& model_filename);
        virtual void SnapshotSolverStateToHDF5(const string& model_filename);
        virtual void RestoreSolverStateFromHDF5(const string& state_filename);
        virtual void RestoreSolverStateFromBinaryProto(const string& state_file);
        
        // history maintains the historical momentum data.
		// update maintains update related data and is not needed in snapshots.
		// temp maintains other information that might be needed in computation
		//   of gradients/updates and is not needed in snapshots
        vector<shared_ptr<Blob<Dtype>>> history_, update_, temp_;
        
        Dtype minimum_loss_;
        int iter_last_event_;
        
        DISABLE_COPY_AND_ASSIGN(SGDSolver);
    };
    
    template <typename Dtype>
    class NesterovSolver: public SGDSolver<Dtype>{
    public:
        explicit NesterovSolver(const SolverParameter& param)
            :SGDSolver<Dtype>(param){}
        explicit NesterovSolver(const string& param_file)
            :SGDSolver<Dtype>(param_file){}
        virtual inline const char* type() const {return "Nesterov";}
        
    protected:
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        
        DISABLE_COPY_AND_ASSIGN(NesterovSolver);
    };
    
    template <typename Dtype>
    class AdaGradSolver: public SGDSolver<Dtype>{
    public:
        explicit AdaGradSolver(const SolverParameter& param)
            :SGDSolver<Dtype>(param){constructor_sanity_check();}
        explicit AdaGradSolver(const string& param_file)
            : SGDSolver<Dtype>(param){constructor_sanity_check();}
        
        virtual inline const char* type() const {return "AdaGrad";}
        
    protected:
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        
        void constructor_sanity_check(){
            CHECK_EQ(0, this->param_.momentum())
                << "Momentum cannot be used with AdaGrad.";
        }
        
        DISABLE_COPY_AND_ASSIGN(AdaGradSolver);
    };
    
    template <typename Dtype>
    class RMSPropSolver: public SGDSolver<Dtype>{
    public:
        explicit RMSPropSolver(const SolverParameter& param)
            :SGDSolver<Dtype>(param){constructor_sanity_check();}
        explicit RMSPropSolver(const string& param_file)
            :SGDSolver<Dtype>(param_file){constructor_sanity_check();}
        virtual inline const char* type() const {return "RMSProp";}
        
    protected:
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        void constructor_sanity_check(){
            CHECK_EQ(0, this->param_.momentum())
                << "Momentum cannot be used with RMSProp.";
            CHECK_GE(this->param_.rms_decay(), 0)
                << "rms_decay should lie between 0 and 1.";
            CHECK_LT(this->param_.rms_decay(), 1)
                << "rms_decay should lie between 0 and 1.";
        }
        
        DISABLE_COPY_AND_ASSIGN(RMSPropSolver);
    };
    
    template <typename Dtype>
    class AdaDeltaSolver: public SGDSolver<Dtype>{
    public:
        explicit AdaDeltaSolver(const SolverParameter& param)
            :SGDSolver<Dtype>(param) {AdaDeltaPreSolve();}
        explicit AdaDeltaSolver(const string& param_file)
            :SGDSolver<Dtype>(param_file) {AdaDeltaPreSolve();}
        
    	virtual inline const char* type() const {return "AdaDelta";}
        
    protected:
        void AdaDeltaPreSolve();
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        
        DISABLE_COPY_AND_ASSIGN(AdaDeltaSolver);
    };
    
    template <typename Dtype>
    class AdamSolver: public SGDSolver<Dtype>{
    public:
        explicit AdamSolver(const SolverParameter& param)
            :SGDSolver<Dtype>(param) {AdamPreSolve();}
        explicit AdamSolver(const string& param_file)
            :SGDSolver<Dtype>(param_file) {AdamPreSolve();}
        virtual inline const char* type() const {return "Adam";}
        
    protected:
        void AdamPreSolve();
        virtual void ComputeUpdateValue(int param_id, Dtype rate);
        
        DISABLE_COPY_AND_ASSIGN(AdamSolver);
    };
    
#endif
}
```

**sgd_solver.cpp**

```c++
#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"

namespace caffe{
    
    // Return the current learning rate. The currently implemented learning rate
    // policies are as follows:
    //    - fixed: always return base_lr.
    //    - step: return base_lr * gamma ^ (floor(iter / step))
    //    - exp: return base_lr * gamma ^ iter
    //    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
    //    - multistep: similar to step but it allows non uniform steps defined by
    //      stepvalue
    //    - poly: the effective learning rate follows a polynomial decay, to be
    //      zero by the max_iter. return base_lr * (1 - iter/max_iter) ^ (power)
    //    - sigmoid: the effective learning rate follows a sigmod decay
    //      return base_lr*( 1/(1 + exp(-gamma * (iter - stepsize))))
    //
    // where base_lr, max_iter, gamma, step, stepvalue and power are defined
    // in the solver parameter protocol buffer, and iter is the current iteration.
    template <typename Dtype>
    Dtype SGDSolver<Dtype>::GetLearningRate(){
        Dtype rate;
        const string& lr_policy = this->param_.lr_policy();
        if(lr_policy == "fixed"){
            rate = this->param_.base_lr();
        }
        else if(lr_policy == "step"){
            this->current_step_ = this->iter_/this->param_.stepsize();
            rate = this->param_.base_lr() * pow(this->param_.gamma(), this->current_step_);
        }
        else if(lr_policy == "exp"){
            rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
        }
        else if(lr_policy == "inv"){
            rate = this->param_.base_lr() * 
                pow(Dtype(1) + this->param_.gamma()*this->iter_,
                   -this->param_.power());
        }
        else if(lr_policy == "multistep"){
            if(this->curent_step_ < this->param_.stepvalue_size() && 
              this->iter_ >= this->param_.stepvalue(this->current_step_)){
                this->current_step_++;
                LOG(INFO) << "MultiStep Statue: Iteration " 
                    << this->iter_ << ", step = " << this->current_step_;
            }
            rate = this->param_.base_lr() * pow(this->param_.gamma(), this->current_step_);
        }
        else if(lr_policy == "poly"){
            rate = this->param_.base_lr() * pow(Dtype(1.)) - 
                (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
            	this->param_.power());
        }
        else if(lr_policy == "sigmoid"){
            rate = this->param_.base_lr() * (Dtype(1) /
                (Dtype(1) + exp(-this->param_.gamma() * (Dtype(this->iter_) - 
                                                        Dtype(this->param_.stepsize())))));
        }
        else{
            LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
        }
        
        return rate;
    }
    
    template <typename Dtype>
    void SGDSolver<Dtype>::PreSolve(){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        
        history_.clear();
        update_.clear();
        temp_.clear();
        
        for(int i=0; i<net_params.size(); ++i){
            const vector<int>& shape = net_params[i]->shape();
            history_.push_back(shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
            update_.push_back(shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
            temp_.push_back(shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
        }
    }
    
    template <typename Dtype>
    void SGDSolver<Dtype>::ClipWeights(){
        const Dtype clip_weights = this->param_.clip_weights();
        if(clip_weights < 0) {return;}
        
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        Dtype sumsq_data = 0;
        for(int i=0; i<net_params.size(); ++i){
            sumsq_data += net_params[i]->sumsq_data();
        }
        
        const Dtype l2norm_data = std::sqrt(sumsq_data);
        if(l2norm_data > clip_weights){
            Dtype scale_factor = clip_weights / l2norm_data;
            LOG(INFO) << "Weight clipping: scaling down weights (L2 norm " 
                << l2norm_data << " > " << clip_weights << ") " 
                << "by scale factor " << scale_factor;
            
            for(int i=0; i<net_params.size(); ++i){
                net_params[i]->scale_data(scale_factor);
            }
        }
    }
    
    template <typename Dtype>
    void SGDSolver<Dtype>::ClipGradients(){
        const Dtype clip_gradients = this->param_.clip_gradients();
        if(clip_gradients < 0) { return; }
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        
        Dtype sumsq_diff = 0;
        for(int i=0; i<net_params.size(); ++i){
            sumsq_diff += net_params[i]->sumsq_diff();
        }
        
        const Dtype l2norm_diff = std::sqrt(sumsq_diff);
        if(l2norm_diff > clip_gradients){
            Dtype scale_factor = clip_gradients / l2norm_diff;
            LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
                << l2norm_diff << " > " << clip_gradients <<  ")"
                << "by scale factor " << scale_factor;
            
            for(int i=0; i<net_params.size(); ++i){
                net_params[i]->scale_diff(scale_factor);
            }
        }
    }
    
   	template <typename Dtype>
    void SGDSolver<Dtype>::ClampWeights(){
        if(!this->param_.has_clamp_weights_lower() && 
           !this->param_.has_clamp_weights_upper()) { return; }
        
        const Dtype lower_bound = this->param_.clamp_weights_lower();
        const Dtype upper_bound = this->param_.clamp_weights_upper();
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        
        for(int i=0; i<net_params.size(); ++i){
            net_params[i]->Clamp(lower_bound, upper_bound);
        }
    }
    
    template <typename Dtype>
    void SGDSolver<Dtype>::ApplyUpdate(){
        Dtype rate = GetLearningRate();
        
        if(this->param_.display() && this->iter_ % this->param_.display() == 0){
            LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << this->iter_
                << ", lr = " << rate;
        }
        
      	ClipGradients();
        for(int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id)
        {
            Normalize(param_id);
            Regularize(param_id);
            ComputeUpdateValue(param_id, rate);
        }
        
        ClipWeights();
        ClampWeights();
        
    	// MSC_VER
        // ........
        
        this->net_->Update();
    }
    
    // 将net中的第param_id个可学习参数的梯度数据缩小 1/iter_size 倍
	// 单次迭代会执行iter_size次的前向和反向过程,每次反向过程都会累加梯度,所以需要先缩小
    template <typename Dtype>
    void SGDSolver<Dtype>::Normalize(int param_id){
        if(this->param_.iter_size() == 1) { return; } //iter_size=1就不用此操作了
        // Scale gradient to counterbalance accumulation.
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params(); //所有可学习参数
        const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size(); // 1/iter_size
        
        switch(Caffe::mode()){
            case Caffe::CPU:{ //cpu模式下,net_params[param_id]的diff_数据全部乘上系数 1/iter_size
                caffe_scal(net_params[param_id]->count(), accum_normalization, 
                          net_params[param_id]->mutable_cpu_diff());
                break;
            }
            case Caffe::GPU:{ //同理,gpu模式下所有参数的diff_也都乘上系数
            #ifndef CPU_ONLY
                caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
                              net_params[param_id]->mutable_cpu_diff());
            #else
                NO_GPU;
            #endif
                break;
            }
            default:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }
    
    // 将网络中的第param_id个参数blob进行l1或l2正则化
    template <typename Dtype>
    void SGDSolver<Dtype>::Regularize(int param_id){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params(); //所有可学习参数
        const vector<float>& net_params_weight_decay = 
            this->net_->params_weight_decay(); //所有可学习参数对应的权重衰减系数
        
        Dtype weight_decay = this->param_.weight_decay(); //求解器参数中设置的基础权重衰减值
        string regularization_type = this->param_.regularization_type(); //求解器参数中设置的正则化类型
        Dtype local_decay = weight_decay * net_params_weight_decay[param_id]; //该参数对应的权重衰减值
        
        switch(Caffe::mode()){
            case Caffe::CPU:{
                if(local_decay){
                    if(regularization_type == "L2"){
                        // add weight decay
        				// L2正则化会在损失函数中增加项 1/2 * λ * θ^2, 因此计算参数的梯度时,每个参数的梯度会增加项 λ * θ
        				// θ对应参数的data_数据, λ对应参数的权重衰减值local_decay
                        caffe_axpy(net_params[param_id]->count(), local_decay,
                                  net_params[param_id]->cpu_data(),
                                  net_params[param_id]->mutable_cpu_diff());//公式 diff_+=local_decay* data_
                    }
                    else if(regularization_type == "L1"){
                        // l1正则化会在损失函数中增加 λ * |θ|, 对应参数的梯度增加 λ * sign(θ)
                        // 判断data_中数据的符号,结果存在临时变量temp_的data_中
                        caffe_cpu_sign(net_params[param_id]->count(),
                                      net_params[param_id]->cpu_data(), 
                                      temp_[param_id]->mutable_cpu_data()); 
                        //公式 diff_ += local_decay * sign(data_)
                        caffe_axpy(net_params[param_id]->count(), local_decay, 
                                  temp_[param_id]->cpu_data(),
                                  net_params[param_id]->mutable_cpu_diff()); 
                    }
                    else{
                        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
                    }
                }
                break;
            }
            case Caffe::GPU:{
            #ifndef CPU_ONLY
                if(local_decay){
                    if(regularization_type == "L2"){
                        // diff_ += local_decay * data_
                        caffe_gpu_axpy(net_params[param_id]->count(), local_decay, 
                                      net_params[param_id]->gpu_data(),
                                      net_params[param_id]->mutable_gpu_diff());
                    }
                    else if(regularization_type == "L1"){
                        //temp_data_ = sign(data_)
                        caffe_gpu_sign(net_params[param_id]->count(),
                                      net_params[param_id]->gpu_data(),
                                      temp_[param_id]->mutable_gpu_data());
                        //diff_ += local_decay * sign(data_)
                        caffe_gpu_axpy(net_params[param_id]->count(), local_decay,
                                      temp_[param_id]->gpu_data(),
                                      net_params[param_id]->mutable_gpu_diff())
                    }
                    else{
                        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
                    }
                }
            #else
                NO_GPU;
            #endif
                break;
            }
            default:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }
    
#ifndef CPU_ONLY
    template <typename Dtype>
    void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate);
#endif
    
    // 根据冲量参数,学习率参数和历史梯度数据,更新当前的梯度值
    template <typename Dtype>
    void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate){
        // 网络中所有参数blob
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        // 每个参数对应的学习率系数
        const vector<float>& net_params_lr = this->net_->params_lr();
        // 求解器参数中设置的冲量
        Dtype momentum = this->param_.momentum();
        // 乘上系数,得到当前参数的学习率
        Dtype local_rate = rate * net_params_lr[param_id];
        
        // Compute the update to history, then copy it to the parameter diff.
        switch(Caffe::mode()){
            case Caffe::CPU:{
                // 计算带冲量的梯度值,并将梯度保存在history_中,供下次迭代使用
                // history_data = local_rate * param_diff + momentum * history_data
                caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                               net_params[param_id]->cpu_diff(), momentum,
                               history_[param_id]->mutable_cpu_data());
                // param_diff = history_data
                caffe_copy(net_params[param_id]->count(),
                          history_[param_id]->cpu_data(),
                          net_params[param_id]->mutable_cpu_diff());
            	break;
            }
            case Caffe::GPU:{
            #ifndef CPU_ONLY
                // 与cpu操作类似,该函数先是 history_data = local_rate * param_diff + momentum * history_data,
    			// 是 param_diff = history_data
                sgd_update_gpu(net_params[param_id]->count(),
                              net_params[param_id]->mutable_gpu_diff(),
                              history_[param_id]->mutable_gpu_data(),
                              momentum, local_rate);
            #else
                NO_GPU;
            #endif
                break;
            }
            default:
                LOG(FATAL) << "Unknown caffe::mod: " << Caffe::mode();
        }
    }
    
    template <typename Dtype>
    void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
      switch (this->param_.snapshot_format()) {
        case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
          SnapshotSolverStateToBinaryProto(model_filename);
          break;
        case caffe::SolverParameter_SnapshotFormat_HDF5:
          SnapshotSolverStateToHDF5(model_filename);
          break;
        default:
          LOG(FATAL) << "Unsupported snapshot format.";
      }
    }
	
    // 将SGDSolver的状态存入SolverState消息中,并存为文件
    template <typename Dtype>
    void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
        const string& model_filename) {
      SolverState state;
      state.set_iter(this->iter_); // 将当前的迭代次数存入SolverState消息中
      state.set_learned_net(model_filename); // 将网络的快照文件名存入
      state.set_current_step(this->current_step_); // 存入迭代的阶段
      state.clear_history(); // 清空历史数据,SolverState消息中的各个参数的历史数据均为BlobProto类型的消息
      for (int i = 0; i < history_.size(); ++i) {
        // Add history
        BlobProto* history_blob = state.add_history(); //增加参数的历史梯度信息
        history_[i]->ToProto(history_blob); // 并将求解器中blob类型history_的数据写入其中
      }
      
      // 生成".solverstate"扩展名的快照状态文件名
      string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
      LOG(INFO)
        << "Snapshotting solver state to binary proto file " << snapshot_filename;
      // 将SolverState消息写入二进制的proto类型文件
      WriteProtoToBinaryFile(state, snapshot_filename.c_str());
    }
	
    // 将SGDSolver的iter_/model_filename/current_step_/history_写入到hdf5文件中
    template <typename Dtype>
    void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
        const string& model_filename) {
      string snapshot_filename =
          Solver<Dtype>::SnapshotFilename(".solverstate.h5"); //先生成文件名
      LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
      hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
          H5P_DEFAULT, H5P_DEFAULT); // 创建hdf5文件
      CHECK_GE(file_hid, 0)
          << "Couldn't open " << snapshot_filename << " to save solver state."; //检查是否创建成功
      hdf5_save_int(file_hid, "iter", this->iter_); // 在file_hid中创建名为"iter"的整形数据集,并将iter_值写入其中
      hdf5_save_string(file_hid, "learned_net", model_filename); // 创建learned_net,并将model_filename写入其中
      hdf5_save_int(file_hid, "current_step", this->current_step_); // 创建"current_step", 并写入
      hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
          H5P_DEFAULT); // 创建"history"组
      CHECK_GE(history_hid, 0)
          << "Error saving solver state to " << snapshot_filename << ".";
      for (int i = 0; i < history_.size(); ++i) {
        ostringstream oss;
        oss << i;
        // 创建Dtype类型的数据集,并将blob中的数据写入其中
        hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
      }
      H5Gclose(history_hid);
      H5Fclose(file_hid);
    }
	
    // 从二进制proto文件state_file中读取求解器的状态,并存入当前求解器中.如果求解器状态中还设置了模型参数文件,则还会加载模型参数
    template <typename Dtype>
    void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
        const string& state_file) {
      SolverState state;
      ReadProtoFromBinaryFile(state_file, &state); // 从state_file文件中读取消息到state中
      this->iter_ = state.iter(); 		// 使用state中的值设置当前的求解器
      if (state.has_learned_net()) { 	// 如果设置了模型参数文件的路径
        NetParameter net_param;
        ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param); // 从文件中读取网络参数
        this->net_->CopyTrainedLayersFrom(net_param); // 数据拷贝至当前网络中
      }
      this->current_step_ = state.current_step(); //设置
      CHECK_EQ(state.history_size(), history_.size())
          << "Incorrect length of history blobs."; //检查state中历史数据的个数与当前求解器中历史数据的个数是否匹配
      LOG(INFO) << "SGDSolver: restoring history";
      for (int i = 0; i < history_.size(); ++i) {
        history_[i]->FromProto(state.history(i)); // 从state中拷贝历史梯度数据至当前求解器中
      }
    }
	
    //从hdf5文件state_file中读取求解器的状态,并存入当前求解器中.如果求解器状态中还设置了模型参数文件,则还会加载模型参数
    template <typename Dtype>
    void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
      hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT); //打开文件
      CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file; //检查操作是否成功
      this->iter_ = hdf5_load_int(file_hid, "iter"); 	//从file_hid中读取"iter"数据集中的整数,存入iter_中
      if (H5LTfind_dataset(file_hid, "learned_net")) {  //判断file_hid中是否存在名为"learned_net"的数据集
        string learned_net = hdf5_load_string(file_hid, "learned_net");
        this->net_->CopyTrainedLayersFrom(learned_net); //读取模型参数文件,加载网络参数
      }
      this->current_step_ = hdf5_load_int(file_hid, "current_step"); //读取"current_step"中的值
      hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT); //打开"history"数据集
      CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
      int state_history_size = hdf5_get_num_links(history_hid);  //获取其中links(元素)的个数
      CHECK_EQ(state_history_size, history_.size())
          << "Incorrect length of history blobs.";  //同样检查是否与当前求解器中的history_匹配
      for (int i = 0; i < history_.size(); ++i) {
        ostringstream oss;
        oss << i;
          
       	// 从history_hid中读取数据,存入history_[i]中
        hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                    kMaxBlobAxes, history_[i].get()); 
      }
      H5Gclose(history_hid);
      H5Fclose(file_hid);
    }

    INSTANTIATE_CLASS(SGDSolver);
    REGISTER_SOLVER_CLASS(SGD);
}
```

 1. 不同的学习策略

    ```c++
    # 参照代码中的GetLearningRate()函数,用Python简单实现了下不同学习率策略的效果,方便有个直观的了解
    import numpy as np
    from math import exp
    import matplotlib.pyplot as plt
    
    base_lr = 0.01
    max_iter = np.arange(3000)
    
    def fixed(iter):
        return base_lr
    
    def step(iter):
        step_size = 500
        gamma = 0.7
        current_step = int(iter / step_size)
        return base_lr * pow(gamma, current_step)
    
    def exp_policy(iter):
        gamma = 0.99
        return base_lr * pow(gamma, iter)
    
    def inv(iter):
        gamma = 0.001
        power = 0.75
        return base_lr * pow(1 + gamma * iter, -power)
    
    class multistep(object):
        gamma = 0.7
        stepvalue = np.array([200, 800, 1500, 2300])
        multistep_current_step = 0
        def rate(self, iter):
            if (self.multistep_current_step < self.stepvalue.shape[0] and
                iter >= self.stepvalue[self.multistep_current_step]):
                self.multistep_current_step += 1
            return base_lr * pow(self.gamma, self.multistep_current_step)
    
    def poly(iter):
        power = 2
        return base_lr * pow(1 - iter / max_iter.shape[0], power)
    
    def sigmoid(iter):
        gamma = -0.01
        step_size = 1500
        return base_lr * (1 / (1 + exp(-gamma * (iter - step_size))))
    
    
    rate_fixed = np.array([fixed(iter) for iter in max_iter])
    rate_step = np.array([step(iter) for iter in max_iter])
    rate_exp_policy = np.array([exp_policy(iter) for iter in max_iter])
    rate_inv = np.array([inv(iter) for iter in max_iter])
    mltstp = multistep()
    rate_multistep = np.array([mltstp.rate(iter) for iter in max_iter])
    rate_poly = np.array([poly(iter) for iter in max_iter])
    rate_sigmoid = np.array([sigmoid(iter) for iter in max_iter])
    
    
    plt.figure(1)
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    ax7 = plt.subplot(3, 3, 7)
    
    plt.sca(ax1)
    ax1.set_title('fixed')
    plt.plot(max_iter, rate_fixed)
    plt.sca(ax2)
    ax2.set_title('step')
    plt.plot(max_iter, rate_step)
    plt.sca(ax3)
    ax3.set_title('exp')
    plt.plot(max_iter, rate_exp_policy)
    plt.sca(ax4)
    ax4.set_title('inv')
    plt.plot(max_iter, rate_inv)
    plt.sca(ax5)
    ax5.set_title('multistep')
    plt.plot(max_iter, rate_multistep)
    plt.sca(ax6)
    ax6.set_title('poly')
    plt.plot(max_iter, rate_poly)
    plt.sca(ax7)
    ax7.set_title('sigmoid')
    plt.plot(max_iter, rate_sigmoid)
    plt.show()
    ```

    ![学习策略](C:\Users\Administrator\Desktop\DrinkCaffe\学习策略.png)

**sgd_solver.cu**

```c++
#include "caffe/util/math_functions.hpp"

namespace caffe{

    template <typename Dtype>
    __global__ void SGDUpdate(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate){
        CUDA_KERNEL_LOOP(i, N){
            g[i] = h[i] = momentum * h[i] + local_rate*g[i];
        }
    }
    
    template <typename Dtype>
    void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate){
        SGDUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, g, h, momentum, local_rate);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template void sgd_update_gpu<float>(int, float*, float*, float, float);
    template void sgd_update_gpu<double>(int, double*, double*, double, double);
}
```

 1. SGD是GD的最基本形式；SGD从权重中减去梯度乘以学习率。momentum动量计算为梯度的移动平均值。
    $$
    v_{i} = \gamma v_{i} + \alpha \frac{\partial L}{\partial \theta_{i} }   \\
    \theta_{i} = \theta_{i}-v_{i}
    $$

​		

**rmsprop_solver.cpp**

```c++
#include <vector>
#include "caffe/sgd_solvers.hpp"

namespace caffe{
    
#ifndef CPU_ONLY
    template <typename Dtype>
    void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay, 
                            Dtype delta, Dtype local_rate);
#endif
    
    template <typename Dtype>
    void RMSPropSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        const vector<float>& net_params_lr = this->net_->params_lr();
        
        Dtype delta = this->param_.delta();
        Dtype rms_decay = this->param_.rms_decay();
        Dtype local_rate = rate * net_params_lr[param_id];
        
        switch(Caffe::mode()){
            case Caffe::CPU:
                caffe_powx(net_params[param_id]->count(), net_params[param_id]->cpu_diff(),
                          Dtype(2), this->update_[param_id]->mutable_cpu_data());
                
                caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1-rms_decay),
                               this->update_[param_id]->cpu_data(), rms_decay,
                               this->history_[param_id]->mutable_cpu_data());
                
                caffe_powx(net_params[param_id]->count(), this->history_[param_id]->cpu_data(),
                          Dtype(0.5), this->update_[param_id]->mutable_cpu_data());
                
                caffe_add_scalar(net_params[param_id]->count(), delta, 
                                this->update_[param_id]->mutable_cpu_data());
                caffe_div(net_params[param_id]->count(), net_params[param_id]->cpu_diff(),
                         this->update_[param_id]->cpu_data(), 
                         this->update_[param_id]->mutable_cpu_data());
                
                caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                               this->update_[param_id]->cpu_data(), Dtype(0),
                               net_params[param_id]->mutable_cpu_diff());
                break;
            case Caffe::GPU:
        #ifndef CPU_ONLY
                rmsprop_update_gpu(net_params[param_id]->count(), 
                                   net_params[param_id]->mutable_gpu_diff(), 
                                  this->history_[param_id]->mutable_gpu_data(),
                                  rms_decay, delta, local_rate);
        #else
                NO_GPU;
        #endif
                break;
            default:
                LOG(FATAL) << "Uknown caffe mode: " << Caffe::mode();
        }
    }
    
    INSTANTIATE_CLASS(RMSPropSolver);
    REGISTER_SOLVER_CLASS(RMSProp);
}
```

**rmsprop_solver.cu**

```c++
#include "caffe/util/math_functions.hpp"

namespace caffe{

	template <typename Dtype>
	__global__ void RMSPropUpdate(int N, Dtype* g, Dtype* h, Dtype rms_decay, Dtype delta, Dtype local_rate)
    {
        CUDA_KERNEL_LOOP(i, N){
            float gi = g[i];
            float hi = h[i] = rms_decay * h[i] + (1 - rms_decay) * gi * gi;
            g[i] = local_rate*g[i] / (sqrt(hi) + delta);
        }
    }
    
    template <typename Dtype>
    void rmsprop_update_gpu(int N, Dtype* g, Dtype* h, Dtype rms_decay, Dtype delta, Dtype local_rate){
        RMSPropUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, g, h, 
                                          						rms_decay, delta, local_rate);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template void rmsprop_update_gpu<float>(int, float*, float*, float, float, float);
    template void rmsprop_update_gpu<double>(int, double*, double*, double, double, double);
}
```

 1. RMSProp自适应地调整每个参数的学习率并允许使用更大的学习率。
    $$
    v_{i} = \beta v_{i} + (1-\beta)\begin{pmatrix}
    \frac{\partial L}{\partial \theta_{i} }
    \end{pmatrix} ^{2}  \\ 
    \theta_{i} = \theta_{i} - \alpha \frac{\frac{\partial L}{\partial \theta_{i} } }{\sqrt{v_{i}} +\varepsilon  }
    $$
    

**nesterov_solver.cpp**

```c++
#include <vector>
#include "caffe/sgd_solvers.hpp"

namespace caffe{
    
#ifndef CPU_ONLY
    template <typename Dtype>
    void nesterov_update_gpu(int N, Dtype* g, Dtpe* h, Dtype momentum, Dtype local_rate);
#endif
    
    template <typename Dtype>
    void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        const vector<float>& net_params_lr = this->net_->params_lr();
        Dtype momentum = this->param_.momentum();
        Dtype local_rate = rate * net_params_lr[param_id];
        
        switch (Caffe::mode()){
            case Caffe::CPU:{
                caffe_copy(net_params[param_id]->count(), this->history_[param_id]->cpu_diff(), 
                           this->update_[param_id]->mutable_cpu_data());
                
                caffe_cpu_axpby(net_params[param_id]->count(), local_rate, 
                               net_params[param_id]->cpu_diff(), momentum, 
                               this->history_[param_id]->mutable_cpu_data());
                
                caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1)+momentum,
                               this->history_[param_id]->cpu_data(), -momentum,
                               this->update_[param_id]->mutable_cpu_data());
                
                caffe_copy(net_params[param_id]->count(), this->update_[param_id]->cpu_data(),
                          net_params[param_id]->mutable_cpu_diff());
                break;
            }
            case Caffe::GPU:{
            #ifndef CPU_ONLY
                nesterov_update_gpu(net_params[param_id]->count(),
                                   net_params[param_id]->mutable_gpu_diff(),
                                   this->history_[param_id]->mutable_gpu_data(),
                                   momentum, local_rate);
            #else
                NO_GPU;
           	#endif
                break;
            }
            default:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }
    
    INSTANTIATE_CLASS(NesteroSolver);
    REGISTER_SOLVER_CLASS(Nesterov);
}
```

**nesterov_solver.cu**

```c++
#include "caffe/util/math_functions.hpp"

namespace caffe{
	
    template <typename Dtype>
    __global__ void NesterovUpdate(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate){
        CUDA_KERNEL_LOOP(i, N){
            float hi = h[i];
            float hi_new = h[i] = momentum * hi + local_rate * g[i];
            g[i] = (1 + momentum) * hi_new - momentum * hi;
        }
    }
    
    template <typename Dtype>
    void nesterov_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate){
        NesterovUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, g, h, 
                                                               	momentum, local_rate);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template void nesterov_update_gpu<float>(int, float*, float*, float, float);
    template void nesterov_update_gpu<double>(int, double*, double*, double, double);
}
```

 1. [Nesterov](https://ocxs.gitbooks.io/deep-learning/content/cs231n/001_training_optimization.html)
    $$
    v^{'} _{i} = v_{i}\\
    
    v_{i} = \gamma v_{i} - \alpha \frac{\partial L}{\partial \theta _{i}}   \\ 
    
    \theta_{i} = \theta_{i} - \gamma  v^{'}_{i} + (1+\gamma) v_{i}
    $$
    

**adam_solver.cpp**

```c++
#include <vector>
#include "caffe/sgd_solvers.hpp"

namespace caffe{
    template <typename Dtype>
    void AdamSolver<Dtype>::AdamPreSolve(){
    	const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        
        for(int i=0; i<net_params.size(); ++i){
            const vector<int>& shape = net_params[i]->shape();
            this->history_.push_back(shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
        }
    }
    
#ifndef CPU_ONLY
    template <typename Dtype>
    void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1,
                        Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate);
#endif
    
    template <typename Dtype>
    void AdamSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        const vector<float>& net_params_lr = this->net_->params_lr();
        Dtype local_rate = rate * net_params_lr[param_id];
        const Dtype beta1 = this->param_.momentum();
        const Dtype beta2 = this->param_.momentum2();
        
        size_t update_history_offset = net_params.size();
        Blob<Dtype>* val_m = this->history_[param_id].get();
        Blob<Dtype>* val_v = this->history_[param_id + update_history_offset].get();
        Blob<Dtype>* val_t = this->temp_[param_id].get();
        
        const int t = this->iter_ + 1;
        const Dtype correction = std::sqrt(Dtype(1) - pow(beta2, t)) / (Dtype(1) - pow(beta1, t));
        const int N = net_params[param_id]->count();
        const Dtype eps_hat = this->param_.delta();
        
        switch(Caffe::mode()){
            case Caffe::CPU:{
                caffe_cpu_axpby(N, Dtype(1)-beta1, net_params[param_id]->cpu_diff(),
                               beta1, val_m->mutable_cpu_data());
                
                caffe_mul(N, net_params[param_id]->cpu_diff(),
                         net_params[param_id]->cpu_diff(), val_t->mutable_cpu_data());
                
                caffe_cpu_axpby(N, Dtype(1)-beta2, val_t->cpu_data(), beta2,
                               val_v->mutable_cpu_data());
                
                caffe_powx(N, val_v->cpu_data(), Dtype(0.5),
                          val_t->mutable_cpu_data());
                
                caffe_add_scalar(N, eps_hat, val_t->mutable_cpu_data());
                
                caffe_div(N, val_m->cpu_data(),
                         val_t->cpu_data(), val_t->mutable_cpu_data());
                
                caffe_cpu_scale(N, local_rate*correction, 
                               val_t->cpu_data(), net_params[param_id]->mutable_cpu_diff());
                break;
            }
            case Caffe::GPU:{
            #ifndef CPU_ONLY
                adam_update_gpu(N, net_params[param_id]->mutable_gpu_diff(),
                               val_m->mutable_gpu_data(), val_v->mutable_gpu_data(), 
                               beta1, beta2, eps_hat, local_rate*correction);
            #else
                NO_GPU;
            #endif
                break;
            }
            defalut:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }
    
    INSATNTIATE_CLASS(AdamSolver);
    REGISTER_SOLVER_CLASS(Adam);
}
```

**adam_solver.cu**

```c++
#include "caffe/util/math_functions.hpp"

namespace caffe{
	template <typename Dtype>
    __global__ void AdamUpdate(int N, Dtype* g, Dtype* m, Dtype* v, 
                              Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate){
        CUDA_KERNEL_LOOP(i, N){
            float gi = g[i];
            float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
            float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
            g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
        }
    }
    
    template <typename Dtype>
    void adam_update_gpu(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1, Dtype beta2,
                        Dtype eps_hat, Dtype corrected_local_rate){
        AdamUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, g, m,
                                                  v, beta1, beta2, eps_hat, corrected_local_rate);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template void adam_update_gpu<float>(int, float*, float*, float*, float, float, float, float);
    template void adam_update_gpu<double>(int, double*, double*, double*, double, double, double, double);
}
```

 1. Adam有点像是momentum和RMSprop的结合，使用平滑的梯度m而不是原来的dx。论文里推荐的参数设置为`eps=1e-8, beta1=0.9, beta2=0.999`. `m`和`v`的初始值设为0
    $$
    m_{i} = \beta _{1}m_{i}+(1-\beta_{1}) \begin{pmatrix}
    \frac{\partial L}{\partial \theta_{i} }
    \end{pmatrix} \\
    
    v_{i} = \beta_{2} v_{i} + (1-\beta_{2})\begin{pmatrix}
    \frac{\partial L}{\partial \theta_{i} }
    \end{pmatrix} ^{2}  \\ 
    
    \theta_{i} = \theta_{i} - \alpha \frac{m_{i} }{\sqrt{v_{i}} +\varepsilon  }
    $$
    

**adagrad_solver.cpp**

```c++
#include <vector>
#include "caffe/sgd_solvers.hpp"

namespace caffe{
#ifndef CPU_ONLY
    template <typename Dtype>
    void adagrad_update_gpu(int N, Dtype* g, Dtype* h, Dtype delta, Dtype local_rate);
#endif
    
    template <typename Dtype>
    void AdaGradSolver<Dtype>::ComputeUpdateValue(int paramid, Dtype rate){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        const vector<float>& net_param_lr = this->net_->params_lr();
        
        Dtype delta = this->param_.delta();
        Dtype local_rate = rate * net_params_lr[param_id];
        
        switch(Caffe::mode()){
            case Caffe::CPU:{
                 // compute square of gradient in update
                caffe_powx(net_params[param_id]->count(),
                          net_params[param_id]->cpu_diff(), Dtype(2),
                          this->update_[param_id]->mutable_cpu_data());
                // update history
                caffe_add(net_params[param_id]->count(),
                          this->update_[param_id]->cpu_data(),
                          this->history_[param_id]->cpu_data(),
                          this->history_[param_id]->mutable_cpu_data());
                // prepare update
                caffe_powx(net_params[param_id]->count(), 
                          this->history_[param_id]->cpu_data(), Dtype(0.5),
                          this->update_[param_id]->mutable_cpu_data());
                
                caffe_add_scalar(net_params[param_id]->count(), delta,
                                this->update_[param_id]->mutable_cpu_data());
                
                caffe_div(net_params[params_id]->count(), 
                         net_params[param_id]->cpu_diff(),
                         this->update_[param_id]->cpu_data(),
                         this->update_[param_id]->mutable_cpu_data());
                // scale and copy
                caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                               this->update_[param_id]->cpu_data(), Dtype(0),
                               net_params[param_id]->mutable_cpu_diff());
                break;
            }
            case Caffe::GPU:{
            #ifndef CPU_ONLY
                adagrad_update_gpu(net_params[param_id]->count(),
                                  net_params[param_id]->mutable_gpu_diff(),
                                  this->history_[param_id]->mutable_gpu_data(),
                                  delta, local_rate);
            #else
                NO_GPU;
            #endif
                break;
            }
            default:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }
    
    INSTANTIATE_CLASS(AdaGradSolver);
    REGISTER_SOLVER_CLASS(AdaGrad);
}

```

**adagrad_solver.cu**

```c++
namespace caffe{
	
    template <typename Dtype>
    __global__ void AdaGradUpdate(int N, Dtype* g, Dtype* h, Dtype delta, Dtype local_rate){
        CUDA_KERNEL_LOOP(i, N){
            float gi = g[i];
            float hi = h[i] = h[i] + gi * gi;
            g[i] = local_rate * gi / (sqrt(hi) + delta);
        }
    }
    
    template <typename Dtype>
    void adagrad_update_gpu(int N, Dtype* g, Dtype* h, Dtype delta, Dtype local_rate){
        AdaGradUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, g, h, delta, local_rate);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template void adagrad_update_gpu<float>(int, float*, float*, float, float);
    template void adagrad_update_gpu<double>(int, double*, double*, double, double);
}
```

 1. Adagrad
    $$
    v_{i} = v_{i} + \begin{pmatrix}
    \frac{\partial L}{\partial \theta _{i}}
    \end{pmatrix} ^{2} \\ 
    
    \theta_{i} = \theta_{i} - \alpha \frac{\frac{\partial L}{\partial \theta _{i}}}{\sqrt{v_{i}} +\varepsilon  } 
    $$
    

**adadelta_solver.cpp**

```c++
#include <vector>
#include "caffe/sgd_solvers.hpp"

    template <typename Dtype>
    void AdaDeltaSolver<Dtype>::AdaDeltaPreSolve(){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        for(int i=0; i<net_params.size(); ++i){
            const vector<int>& shape = net_params[i]->shape();
            this->history_.push_back(shared_ptr<Blob<Dtype>>(new Blob<Dtype>(shape)));
        }
    }

    #ifndef CPU_ONLY
    template <typename Dtype>
    void adadelta_update_gpu(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum, 
                            Dtype delta, Dtype local_rate);
    #endif

    template <typename Dtype>
    void AdaDeltaSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate){
        const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
        const vector<float>& net_params_lr = this->net_->params_lr();

        Dtype delta = this->param.delta();
        Dtype momentum = this->param.momentum();
        Dtype local_rate = rate * net_params_lr[param_id];
        size_t update_history_offset = net_params.size();

        switch(Caffe::mode()){
            case Caffe::CPU：{
                // compute square of gradient in update
                caffe_powx(net_params[param_id]->count(),
                          net_params[param_id]->cpu_diff(), Dtype(2), 
                          this->update_[param_id]->mutable_cpu_data());
                // update history of gradients
                caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1)-momentum,
                               this->update_[param_id]->cpu_data(), momentum,
                               this->history_[param_id]->mutable_cpu_data());
                // add delta to history to guard against dividing by zero later
                caffe_set(net_params[param_id]->count(), delta, 
                         this->temp_[param_id]->mutable_cpu_data());

                caffe_add(net_params[param_id]->count(), this->temp_[param_id]->cpu_data(),
                         this->history_[update_history_offset + param_id]->cpu_data(),
                         this->update_[param_id]->mutable_cpu_data());
                caffe_add(net_params[param_id]->count(), this->temp_[param_id]->cpu_data(),
                         this->history_[param_id]->cpu_data(), 
                         this->temp_[param_id]->mutable_cpu_dara());

                // divide history of updates by history of gradients
                caffe_div(net_params[param_id]->count(), this->update_[param_id]->cpu_data(),
                          this->temp_[param_id]->cpu_data(), 
                          this->update_[param_id]->mutable_cpu_data());
                // jointly compute the RMS of both for update and gradient history
                caffe_powx(net_params[param_id]->count(), this->update_[param_id]->cpu_data(),
                          Dtype(0.5), this->update_[param_id]->mutable_cpu_data());

                // compute the update
                caffe_mul(net_params[param_id]->count(), net_params[param_id]->cpu_diff(),
                         this->update_[param_id]->cpu_data(), 
                         net_params[param_id]->mutable_cpu_diff());
                // compute square of update
                caffe_powx(net_params[param_id]->count(), 
                          net_params[param_id]->cpu_diff(), Dtype(2),
                          this->update_[param_id]->mutable_cpu_data());
                // update history of updated
                caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1)-momentum,
                               this->update_[param_id]->cpu_data(), momentum,
                               this->update_[update_history_offset + param_id]->mutable_cpu_data());
                // apply learning rate
                caffe_cpu_scale(net_params[param_id]->count(), local_rate,
                               net_params[param_id]->cpu_diff(),
                               net_params[param_id]->mutable_cpu_diff());
                break;
            }
            case Caffe::GPU:{
            #ifndef CPU_ONLY
                adadelta_update_gpu(net_params[param_id]->count(),
                                   net_params[param_id]->mutable_gpu_diff(),
                                   this->history_[param_id]->mutable_gpu_data(),
                                   this->history_[update_history_offset + param_id]->mutable_gpu_data(),
                                   momentum, delta, local_rate);
            #else
                NO_GPU;
            #endif
                break;
            }    

            default:
                LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
    }
	
	INSATNTIATE_CLASS(AdaDeltaSolver);
	REGISTER_SOLVER_CLASS(AdaDelta);
}
```

**adadelta_solver.cu**

```c++
#include "caffe/util/math_functions.hpp"

namespace caffe{
	
    template <typename Dtype>
    __gobal__ void AdaDeltaUpdate(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum, 
                                 Dtype delta, Dtype local_rate){
        CUDA_KERNEL_LOOP(i, N){
            float gi = g[i];
            float hi = h[i] = momentum * h[i] + (1 - momentum) * gi * gi;
            gi = gi * sqrt((h2[i] + delta) / (hi + delta));
            h2[i] = momentum * h2[i] + (1 - momentum) * gi * gi;
            g[i] = local_rate * gi;
        }
    }
    
    template <typename Dtype>
    void adadelta_update_gpu(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum, 
                            Dtype delta, Dtype local_rate){
        AdaDeltaUpdate<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        			N, g, h, h2, momentum, delta, local_rate);
        CUDA_POST_KERNEL_CHECK;
    }
    
    template void adadelta_update_gpu<float>(int, float*, float*, float*, float, float, float);
    template void adadelta_update_gpu<double>(int, double*, double*, double*, double, double, double);
}
```

 	1.  [AdaDelta](http://zh.gluon.ai/chapter_optimization/adadelta.html)
