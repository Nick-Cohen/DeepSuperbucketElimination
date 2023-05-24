#ifndef FunctionNN_HXX_INCLUDED
#define FunctionNN_HXX_INCLUDED

#ifdef WINDOWS
#ifndef NOMINMAX
 #define NOMINMAX
#endif
#include "windows.h"
#endif

#include <climits>
#include <stdlib.h>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>
#include <fstream>

#include "Utils/Mutex.h"
#include "Utils/Sort.hxx"
#include "Problem/Globals.hxx"
#include "Problem/Function.hxx"
#include "Problem/Workspace.hxx"
#include "Net.h"
#include "DATA_SAMPLES.h"
#include "NNConfig.h"

#ifdef INCLUDE_TORCH
#include <torch/torch.h>
#endif // INCLUDE_TORCH

namespace BucketElimination { class Bucket ; class MiniBucket ; }

namespace ARE
{

class ARP ;

class FunctionNN : public Function
{

private:

   DATA_SAMPLES * DS = NULL;
   Config config;

#ifdef INCLUDE_TORCH
   //Net* model = NULL;//NOTE this one was auto in all the turorials   //TODO ask KALEV if this one can be nargs
   Masked_Net * model = NULL;
#endif // INCLUDE_TORCH

//   torch::Tensor empty_tensor = torch::empty(_nArgs);
//   torch::DeviceType device_type_inf = torch::kCPU;
//   torch::Device device_inf(device_type_inf);

public :

	float ln_max_value = (std::numeric_limits<float>::min)();
    float ln_sum = 0.0;
    float sum_ln = 0.0;
    float sum_ln_abs = 0.0;
    float ln_min_value = (std::numeric_limits<float>::max)();
    float local_min_e=0,local_max_e=0,local_avg_e=0,local_w_min_e=0,local_w_max_e=0,local_w_avg_e=0,local_var_e=0;

    int32_t train_samples = 0;
    int32_t val_samples = 0;
    int32_t test_samples = 0;

	virtual int32_t AllocateTableData(void)
	{
		// NO TABLE!!!
		return 0 ;
	}

	virtual ARE_Function_TableType TableEntryEx(int32_t *BEPathAssignment, const int32_t *K) const 
	/*
		desc = return fn value corresponding to given input configuration...
		BEPathAssignment = assignment to all variables on the path from the bucket to the root of the bucket tree...
	    this fn is a wrapper for these two lines... when functions are table-based...
		adr = fn->ComputeFnTableAdr_wrtLocalPermutation(BEPathAssignment, K) ;
		double v = fn->TableEntry(adr) ;
		which is 
		return fn->TableEntry(fn->ComputeFnTableAdr_wrtLocalPermutation(BEPathAssignment, K)) ;
		OVERWRITE in a NN-based fn...
	*/
	{
		ARE_Function_TableType out_value = 0.0 ;
#ifdef INCLUDE_TORCH

       // printf("\n in function table ----");
        bool isNet;
        bool exp_converted=true;
        isNet = 0 == global_config.network.compare("net");
        if(isNet){
            exp_converted=false;
            //if(global_config.l==1)
            //{
            //    exp_converted=true;
            //}
        }

        auto start = std::chrono::high_resolution_clock::now();
        torch::DeviceType device_type = torch::kCPU;
        torch::Device device(device_type);

        model->to(device);
        model->eval();

        auto empty_tensor = torch::empty(_nArgs);
        float* myData = empty_tensor.data_ptr<float>();
//        printf("******************************\n");
//        printf("BEPathAssignment:\n");
//        for (int i=0; i<_nArgs; i++){
//            printf("%d ", BEPathAssignment[i]);
//        }
//        printf("\n_ArgumentsPermutationList:\n");
//        for (int i=0; i<_nArgs; i++){
//            printf("%d ", _ArgumentsPermutationList[i]);
//        }
////        printf("\nBEPathAssignment[_ArgumentsPermutationList[i]:\n");
//        for (int i=0; i<_nArgs; i++){
//            printf("%d ", BEPathAssignment[_ArgumentsPermutationList[i]]);
//        }
//        printf("\n");

        for (int i=0; i <_nArgs; i++)
            *myData++ = (float)BEPathAssignment[_ArgumentsPermutationList[i]];//(float)BEPathAssignment[_ArgumentsPermutationList[i]]

        torch::Tensor input = empty_tensor.resize_(_nArgs).clone();
        input = input.to(device);
        auto output = model->forward(input,true,isNet);

        double out_value;

        if(global_config.do_sum==2)
        {
            out_value = (double)output.x.item<double>()+ln_sum;
        }
        if(global_config.do_sum==1)
        {
            out_value = (double)output.x.item<double>();
            out_value = ln_min_value + out_value*(ln_max_value - ln_min_value);
            //out_value = log(out_value) + ln_sum;
        }
        else if (global_config.do_sum==-1){
            out_value = (double)output.x.item<double>();
            out_value = ln_min_value + (out_value+1)*(ln_max_value - ln_min_value)/2;
        }
        else if(global_config.do_sum==0){
            out_value = (double)output.x.item<double>()*ln_max_value;
        }
        else if(global_config.do_sum==3){
        out_value = (double)output.x.item<double>()*(sum_ln) + ln_min_value;
        }
       // torch::Tensor output = model->forward(input);
        if(exp_converted){
            //printf("calculating log_value here ----");
            out_value = log(out_value);
        }
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        double duration = microseconds*1.0 / 1000000;
        this->WS()->time_TableEntryEx += duration;
        this->WS()->count_TableEntryEx++;
        //rintf("\n out of functiontable ----");

#endif // INCLUDE_TORCH
		return out_value ;
	}

public :

	virtual void Initialize(Workspace *WS, ARP *Problem, int32_t IDX)
	{
		Function::Initialize(WS, Problem, IDX) ;
		//here we assume the structure is known just initialize the most simple FF neural network with no training.
	}

	void Destroy(void)
	{
		// TODO own stuff here...
		Function::Destroy() ;

	}

	void After_Train(void)
    {
#ifdef INCLUDE_TORCH
		torch::DeviceType device_type = torch::kCPU;
		torch::Device device(device_type);
		model->to(device);
		model->eval();
#endif // INCLUDE_TORCH
    }

  //  torch::Tensor weighted_mse_loss(auto input,auto target,auto weight);

#ifdef INCLUDE_TORCH
    void confusion_matrix(torch::Tensor & prediction, torch::Tensor & truth, int & true_positives_batch, int & false_positives_batch, int & true_negatives_batch, int & false_negatives_batch)
	{
		torch::Tensor confusion_vector = prediction / truth;
		true_positives_batch = (confusion_vector==1).sum().template item<int>();
		false_positives_batch = (confusion_vector== float('inf')).sum().template item<int>();
		true_negatives_batch = isnan(confusion_vector).sum().template item<int>();
		false_negatives_batch = (confusion_vector==0).sum().template item<int>();
	}
#endif // INCLUDE_TORCH

        void Train(DATA_SAMPLES *DS_train, DATA_SAMPLES *DS_val, bool to_save,int bucket_num)
        {
		printf("in training --- ----");

#ifdef INCLUDE_TORCH

	    auto start = std::chrono::high_resolution_clock::now();
        //for we need to read the samples from the table!
//        DS->print_data();
        _TableData = NULL;
        _TableSize = 0;
        bool isNet=false;
        if (0 == global_config.network.compare("net")){
            isNet=true;
        }
        if (global_config.var_dim>=1)
            global_config.h_dim = int(_nArgs*global_config.var_dim);
        /*
        if(global_config.vareps==1) {
            if (_nArgs < 75)
                global_config.e = 1e-6;
            else if (_nArgs < 120)
                global_config.e = 1e-6;
        }
        else {
            if (_nArgs < 75)
                global_config.e = 1e-7;
            else if (_nArgs < 120)
                global_config.e = 1e-6;
        }
        */
        if(global_config.epsilon>0){
            if (global_config.prev_e>0)
                global_config.e = 1e-8;
        }
        if (global_config.prev_e>0)
            global_config.e = 1e-8;

        if (model == NULL)
            model = new Masked_Net(_nArgs);

        Masked_Net * w_model = new Masked_Net(_nArgs);

        torch::DeviceType device_type;
        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device_type =  torch::kCUDA;//kCUDA
        } else {
            std::cout << "Training on CPU." << std::endl;
            device_type = torch::kCPU;
        }

        torch::Device device(device_type);
        w_model->to(device);
        w_model->train();

        auto dataset_train = DS_train->map(torch::data::transforms::Stack<>());
        int64_t batch_size = global_config.batch_size;
        auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset_train,batch_size);

        auto dataset_val = DS_val->map(torch::data::transforms::Stack<>());
        auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset_val,batch_size);

        torch::optim::Adam optimizer(w_model->parameters(), torch::optim::AdamOptions(global_config.lr, global_config.e));
        //torch::optim::SGD optimizer(w_model->parameters(),torch::optim::SGDOptions(global_config.lr));
        int64_t n_epochs = global_config.n_epochs;
        printf("%d", batch_size);
        float best_mse = std::numeric_limits<float>::max();
        float mse=0, val_mse=0, prev_val_mse=std::numeric_limits<float>::max();
        float w_mse=0, w_val_mse=0,mse_to_compare=0, c_loss=0, val_c_loss=0, train_rel_error=0, val_rel_error=0, test_rel_error=0;
        int count=0, epoch,epoch_t=0;
        bool isTest;
        float zero = 0.0;
        float non_zeros =0.0,train_non_zeros=0.0, val_non_zeros=0.0;
        int total_batch =0, neg_batch =0, neg=0;
        int true_positives=0, false_positives=0, true_negatives=0, false_negatives=0;
        int true_positives_batch=0, false_positives_batch=0, true_negatives_batch=0, false_negatives_batch=0;
        int true_positives_t=0, false_positives_t=0, true_negatives_t=0, false_negatives_t=0;
        int prev_false_negatives=10000,best_false_negatives=10000, fn_to_compare=10000;
        float val_l_t=0. ,  w_val_l_t =0., loss_to_compare=0.,best_l2_reg, test_re, train_re, val_re, first_loss=0 ;
        torch::Tensor val_loss_total, val_w_loss_total, w_loss_total, loss_total,ln_out, ln_target, w, l2_reg, l1_reg, rel_error;
        std::string o_file_b = global_config.out_file + "loss.txt";
        std::ofstream to_write_b;
        to_save=false;

        int t=0,b=0;
        float effective_N = 0;

        for (epoch = 1; epoch <= n_epochs; epoch++) {
            printf("epoch %d",epoch);
            isTest=false;
            size_t batch_idx = 0;
            size_t val_batch_idx = 0;
            mse = 0.; // mean squared error
            val_mse = 0.;
            c_loss = 0.;
            val_c_loss = 0.;
            w_mse = 0.; // mean squared error
            w_val_mse = 0.;
            train_re=0.0;
            val_re=0.0;
            //int count = 0;

            for (auto &batch : *data_loader_train) {
                effective_N = 0;
                auto imgs = batch.data;
                torch::Tensor loss, w_loss, loss_n;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(device);
                labels = labels.to(device);
                optimizer.zero_grad();

                //auto output = w_model->forward(imgs);
                auto output = w_model->forward(imgs,isTest,isNet);

                if(to_save) {
                    if(t==0){
                        to_write_b.open(o_file_b, std::ios_base::app);
                        if (to_write_b.is_open()) {
                            //printf("writing to the file %s", o_file_b.c_str());
                            if (isNet)
                                to_write_b << _nArgs << '\t' << labels.mean().template item<float>() << '\t';
                            to_write_b.close();
                        }
                    t=1;
                    }
                }
                if (global_config.do_sum==0){
                    //w = exp(labels*ln_max_value-ln_sum);
                    if (global_config.do_weight==0)
                        w = train_samples*labels/(sum_ln_abs*ln_max_value);
                    else
                        w = labels/(sum_ln_abs*ln_max_value);
                }
                else if (global_config.do_sum==1){
                    if (global_config.do_weight==0){
                        w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln_abs*ln_max_value))*train_samples;
                    }
                    else{
                        //w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln_abs*ln_max_value));
                        //w = exp(ln_min_value + labels*(ln_max_value - ln_min_value) -ln_sum)*train_samples;
                        w = (( labels*(ln_max_value - ln_min_value))/ (sum_ln));

                    }
                }
                else if (global_config.do_sum==-3){
                    if (global_config.do_weight==0) {
                        w = train_samples*((ln_min_value + (labels+1) * (ln_max_value - ln_min_value)) / (2*(sum_ln_abs * ln_max_value)));
                    }
                    else
                        w = ((ln_min_value + (labels+1) * (ln_max_value - ln_min_value)) / (2*(sum_ln_abs * ln_max_value)));
                }
                else if (global_config.do_sum==2){
                    w = exp(labels);
                }
                else
                    w = labels;
                if (isNet){
                    if(0==global_config.loss.compare("wmse")){
                        if (global_config.l2_loss==1)
                            w_loss = (w*((output.x - labels).pow(2))).mean();
                        else
                            w_loss = (w*torch::abs((output.x - labels))).mean();
                            //w_loss = (ln_max_value-ln_min_value)*(w*torch::abs((output.x - labels))).mean();

                        if (global_config.l2_loss==1)
                            loss = torch::nn::functional::mse_loss(labels, output.x);
                        else
                            loss = torch::abs((output.x - labels)).mean();
                            //loss = (ln_max_value-ln_min_value)*torch::abs((output.x - labels)).mean();
                        w_loss_total = w_loss;
                        loss_total = loss;
                        //rel_error = torch::abs((output.x-labels).div(labels)).mean();
                        rel_error = (torch::abs((output.x-labels)/labels)).mean();
                        //std::cout<<"Calculating loss --";
                        //std::cout<<l2_reg;


                        //std::cout<<l2_reg;
                            //if(global_config.l2==0)
                            //    w_loss_total = w_loss_total + global_config.l2_lambda * l1_reg;
                            //else
                        if (global_config.reg>0)
                        {
                            l2_reg = torch::zeros(1).cuda();
                            l1_reg = torch::zeros(1).cuda();
                            for (const auto& p : w_model->parameters()) {
                                l2_reg = l2_reg +  torch::norm(p);
                                l1_reg = l1_reg + torch::sum(torch::abs(p));
                            }
                            if(global_config.l2==0)
                            {
                                w_loss_total = w_loss_total + global_config.l2_lambda * l1_reg;
                                loss_total = loss_total + global_config.l2_lambda * l1_reg;
                            }
                            else{
                                w_loss_total = w_loss_total + global_config.l2_lambda * l2_reg;
                                loss_total = loss_total + global_config.l2_lambda * l2_reg;
                            }
                                //std::cout<<"l2 regularization ----";
                        }

                        //l1_reg =  torch::zeros(1).cuda();
                        //std::cout<<"Calculating loss 2--";
                        //
                    }
                    else{
                       // printf("calculating loss -- ");
                        ln_target = log(w);
                        ln_out = log(output.x);
                        std::cout<< output.x << log(output.x);
                        w_loss = (w*(ln_target-ln_out)).mean();
                        loss = (ln_target - ln_out).mean();
                        //std::cout << w << ln_out << w_loss << loss;
                        w_loss_total = w_loss;
                        loss_total = loss;

                        //float a = w_loss.template item<float>();
                       // printf("loss - %f",w_loss.template item<float>());
                    }
                }
                else{
                    float w_max = w.max().template item<float>();
                    auto label_binary = torch::where(labels>0, torch::ones_like(labels), torch::zeros_like(labels));
                    loss_n =  torch::binary_cross_entropy(output.masked,label_binary);
                    non_zeros = (label_binary==1).sum().template item<float>();

                    //l2_reg = torch::zeros(1).cuda();
                    //for (const auto& p : w_model->parameters()) {
                    //    l2_reg = l2_reg +  torch::norm(p);
                        //    l1_reg = l1_reg + torch::sum(torch::abs(p));
                    //}
                    //std::cout<<l2_reg;
                    //if(global_config.l2==0)
                    //    w_loss_total = w_loss_total + global_config.l2_lambda * l1_reg;
                    //else
                    w_loss_total = global_config.loss_weight_mse*(w*((output.x - labels).pow(2))).mean() + global_config.loss_weight *loss_n;
                    loss_total = global_config.loss_weight_mse*torch::nn::functional::mse_loss(labels, output.x) + global_config.loss_weight *loss_n;
                    if (global_config.reg>0){
                        w_loss_total = w_loss_total + global_config.l2_lambda * l2_reg;
                        loss_total = loss_total + global_config.l2_lambda * l2_reg;
                    }

                    loss = torch::nn::functional::mse_loss(labels, output.x);
                    w_loss = (w*((output.x - labels).pow(2))).mean();
                    neg_batch = (label_binary==0).sum().template item<float>();
                    neg = neg + neg_batch;
                    confusion_matrix(output.masked, label_binary, true_positives_batch, false_positives_batch, true_negatives_batch, false_negatives_batch);
                    true_positives_t += true_positives_batch;
                    false_positives_t += false_positives_batch;
                    true_negatives_t += true_negatives_batch;
                    false_negatives_t += false_negatives_batch;

                    rel_error = (torch::abs((output.x-labels)/labels)).mean();
                }
               //auto loss = torch::nn::functional::mse_loss(labels, output);

                if(0 == global_config.s_method.compare("is")){
                    w_loss_total.backward();
                    first_loss = w_loss_total.template item<float>();
                    //std::cout<<"\n w_loss"<< w_loss_total;
                }
                else{
                    loss_total.backward();
                    first_loss = loss_total.template item<float>();
                    //std::cout<<"loss"<< loss_total;
                }
                optimizer.step();

                effective_N += (torch::sum(w).square()/torch::sum(w.square())).template item<float>();

                /*
                auto state = optimizer;
                auto step_size = state[""];
                auto& exp_avg = state.exp_avg();
                auto& exp_avg_sq = state.exp_avg_sq();
                auto& max_exp_avg_sq = state.max_exp_avg_sq();
                */

                float l = loss_total.template item<float>();
                float w_l = w_loss_total.template item<float>();
                float a = rel_error.template item<float>();

                mse += l;
                w_mse += w_l;
                train_re += a;

                if (!isNet){
                    float c_l = loss_n.template item<float>();
                    c_loss += c_l;
                    train_non_zeros += non_zeros;
                }
                //printf("batch_index : %d, mse: %f", batch_idx, mse);
                batch_idx++;
            }


            //count++;

            printf("\t  %d %d %d %d \t ",true_positives_t, false_positives_t, true_negatives_t, false_negatives_t);
            mse /= (float) batch_idx;
            w_mse /= (float) batch_idx;
            train_re /= (float) batch_idx;

            //first_loss=mse;
            printf("Train relative error ------- %f \n",train_re);

            if(to_save){
                if(b==0){
                    to_write_b.open(o_file_b,std::ios_base::app);
                    if (to_write_b.is_open())
                    {
                        //printf("writing to the file %s", o_file_b.c_str());
                        if (isNet)
                            to_write_b << first_loss << '\t' ;
                        to_write_b.close();
                    }
                    b=1;
                }
            }

            for (auto& group : optimizer.param_groups()) {
                for (auto& p : group.params()) {
                    if (!p.grad().defined())
                        continue;

                    auto grad = p.grad();
                    float g = grad.mean().template item<float>();
                    //std::cout<<" writing gradient ----- \n"<<g;
                    /*
                    if(to_save){
                        to_write_b.open(o_file_b,std::ios_base::app);
                        if (to_write_b.is_open())
                        {
                            //printf("writing to the file %s", o_file_b.c_str());
                            if (isNet)
                                to_write_b << float(g) << '\t';
                            to_write_b.close();
                        }
                    }
                     */
                }

                //exit(1);
                auto &options = static_cast<torch::optim::AdamOptions &>(group.options());
                double current_lr = options.lr();
                double eps = options.eps();
                printf(" current eps ---- %f \n", eps);
                //std::cout<< options.eps();
            }

            if (!isNet)
            {
                c_loss /= (float) batch_idx;
                printf("Non zeros in training set -- %d, cross-entropy loss -- %f", train_non_zeros, c_loss);
            }

            printf("Epoch number : %d, train mse : %f train wmse : %f", epoch, mse,w_mse );
            // printf("Mean squared error for training data: %f\n", mse);
            torch::Tensor val_loss, w_val_loss, val_loss_n;
            true_positives=0, false_positives=0, true_negatives=0, false_negatives=0;
            isTest=true;

            batch_idx = 0;
            for (auto &batch : *data_loader_val) {
                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(device);
                labels = labels.to(device);
                auto output = w_model->forward(imgs,isTest,isNet);
                if (global_config.do_sum==0) {
                    if (global_config.do_weight == 0)
                        w = val_samples * labels / sum_ln_abs;
                    else
                        w = labels / sum_ln_abs;
                }
                else if (global_config.do_sum==2)
                    w = exp(labels);
                else if (global_config.do_sum==1){
                    if (global_config.do_weight==0) {
                        w = ((ln_min_value + labels * (ln_max_value - ln_min_value)) / (sum_ln_abs * ln_max_value)) *
                            val_samples;
                    }
                    else{
                        //w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln_abs*ln_max_value));
                        //w = exp(ln_min_value + labels*(ln_max_value - ln_min_value) -ln_sum)*val_samples;
			//w = exp((ln_min_value + labels * (ln_max_value - ln_min_value)) - ln_sum)*val_samples;

			w = (( labels*(ln_max_value - ln_min_value))/ (sum_ln));
                    }
                }
                else if (global_config.do_sum==-3){
                    if (global_config.do_weight==0) {
                        w = val_samples*((ln_min_value + (labels+1) * (ln_max_value - ln_min_value)) / (2*(sum_ln_abs * ln_max_value)));
                    }
                    else
                        w = ((ln_min_value + (labels+1) * (ln_max_value - ln_min_value)) / (2*(sum_ln_abs * ln_max_value)));
                }
                else
                    w = labels;

                if(isNet){
                    if(0==global_config.loss.compare("wmse")){
                        //w_val_loss = (w*((output.x - labels).pow(2))).mean();
                        //val_loss = torch::nn::functional::mse_loss(labels, output.x);
                        if (global_config.l2_loss==1)
                            w_val_loss = (w*((output.x - labels).pow(2))).mean();
                        else
                            w_val_loss = (w*torch::abs((output.x - labels))).mean();
                            // w_val_loss = (ln_max_value-ln_min_value)*(w*torch::abs((output.x - labels))).mean();
                        if (global_config.l2_loss==1)
                            val_loss = torch::nn::functional::mse_loss(labels, output.x);
                        else
                            val_loss = torch::abs((output.x - labels)).mean();
                            //val_loss = (ln_max_value-ln_min_value)*torch::abs((output.x - labels)).mean();
                       // std::cout<<"\n" << val_loss<<w_val_loss;
                        w_loss_total = w_val_loss;
                        loss_total = val_loss;
                        rel_error = (torch::abs((output.x-labels)/labels)).mean();

                        if (global_config.reg>0){
                            l2_reg = torch::zeros(1).cuda();
                            l1_reg = torch::zeros(1).cuda();
                            for (const auto& p : w_model->parameters()) {
                                l2_reg = l2_reg +  torch::norm(p);
                                l1_reg = l1_reg + torch::sum(torch::abs(p));
                            }
                            //if(global_config.l2==0)
                            //    w_loss_total = w_loss_total + global_config.l2_lambda * l1_reg;
                            //else

                            if(global_config.l2==0){
                               w_loss_total = w_loss_total + global_config.l2_lambda * l1_reg;
                               loss_total = loss_total + global_config.l2_lambda * l1_reg;
                            }
                            else{
                                //std::cout<<"l2 regularization --";
                                w_loss_total = w_loss_total + global_config.l2_lambda * l2_reg;
                                loss_total = loss_total + global_config.l2_lambda * l2_reg;
                            }
                        }
                        //l2_reg = torch::zeros(1).cuda();
                        //l1_reg = torch::zeros(1).cuda();
                        //for (const auto& p : w_model->parameters()) {
                        //    l2_reg = l2_reg + torch::norm(p);
                        //    l1_reg = l1_reg + torch::sum(torch::abs(p));
                        //}
                        //if(global_config.l2==0)
                        //    w_val_loss = w_val_loss + global_config.l2_lambda * l1_reg;
                        //else
                        //    w_val_loss = w_val_loss + global_config.l2_lambda * l2_reg;
                    }
                    else{
                        ln_target = log(w);
                        ln_out = log(output.x);
                        std::cout<< output.x << log(output.x);
                        w_val_loss = (w*(ln_target-ln_out)).mean();
                        val_loss = (ln_target - ln_out).mean();
                        //std::cout << w << ln_out << w_loss << loss;
                        w_loss_total = w_val_loss;
                        loss_total = val_loss;
                    }
                }
                else{
                    auto label_binary = torch::where(labels>0, torch::ones_like(labels), torch::zeros_like(labels));
                    val_loss_n =  torch::binary_cross_entropy(output.masked,label_binary);
                    val_c_loss += val_loss_n.template item<float>();
                    //non_zeros = torch.count_nonzero(label_binary);
                    non_zeros = (label_binary==1).sum().template item<float>();
                    val_non_zeros += non_zeros;
                    total_batch = label_binary.sizes()[0];
                    neg_batch = (label_binary==0).sum().template item<float>();
                    neg_batch = total_batch - non_zeros;
                    neg = neg + neg_batch;
                    confusion_matrix(output.masked, label_binary, true_positives_batch, false_positives_batch, true_negatives_batch, false_negatives_batch);
                    val_w_loss_total = global_config.loss_weight_mse*(w*((output.x - labels).pow(2))).mean() + global_config.loss_weight *val_loss_n;
                    val_loss_total = global_config.loss_weight_mse*torch::nn::functional::mse_loss(labels, output.x) + global_config.loss_weight *val_loss_n;
                    //std::cout<<l2_reg;
                    //if(global_config.l2==0)
                    //    w_loss_total = w_loss_total + global_config.l2_lambda * l1_reg;
                    //else

                    if (global_config.reg>0){
                        l2_reg = torch::zeros(1).cuda();
                        for (const auto& p : w_model->parameters()) {
                            l2_reg = l2_reg +  torch::norm(p);
                            //    l1_reg = l1_reg + torch::sum(torch::abs(p));
                        }
                        val_w_loss_total  = val_w_loss_total + global_config.l2_lambda * l2_reg;
                        val_loss_total = val_loss_total + global_config.l2_lambda * l2_reg;
                    }
                    val_loss = torch::nn::functional::mse_loss(labels, output.x);
                    printf("%f %f",val_loss.template item<float>(),val_loss_n.template item<float>() );
                    //val_w_loss_total = val_loss_n;
                    //val_loss_total = val_loss_n;
                    true_positives += true_positives_batch;
                    false_positives += false_positives_batch;
                    true_negatives += true_negatives_batch;
                    false_negatives += false_negatives_batch;
                    printf("\t  %d %d %d %d \t ",true_positives, false_positives, true_negatives, false_negatives);

                    w_val_loss = (w*((output.x - labels).pow(2))).mean();
                    val_loss = torch::nn::functional::mse_loss(labels, output.x);
                    rel_error = (torch::abs((output.x-labels)/labels)).mean();
                }

                val_mse += loss_total.template item<float>();
                w_val_mse += w_loss_total.template item<float>();
                val_re += rel_error.template item<float>();

                if (!isNet){
                    val_l_t += val_loss_total.template item<float>();
                    w_val_l_t += val_w_loss_total.template item<float>();
                }
                batch_idx++;
                val_batch_idx++;
            }

            val_mse /= (float) val_batch_idx ;
            w_val_mse /= (float) val_batch_idx;
            val_re /= (float) val_batch_idx;
            printf("Validation relative error ------- %f \n",val_re);
            /*
            if(to_save){
                to_write_b.open(o_file_b,std::ios_base::app);

                if (to_write_b.is_open())
                {
                    //printf("writing to the file %s", o_file_b.c_str());
                    if (isNet)
                        to_write_b << val_re << '\t' << ln_max_value << '\t' << ln_min_value << '\t' << val_re*(ln_max_value-ln_min_value) << '\n' ;
                    to_write_b.close();
                }
            }
            */
            if (!isNet) {
                val_l_t /= (float) val_batch_idx;
                w_val_l_t /= (float) val_batch_idx;
            }
            printf("\t  %d %d %d %d \t ",true_positives, false_positives, true_negatives, false_negatives );

            if (!isNet){
                val_c_loss /= (float) val_batch_idx;
                printf("Non zeros in validation set -- %d & val cross-entropy loss -- %f", val_non_zeros,val_c_loss);
            }
            printf("Validation set error : %f, %f", val_mse,w_val_mse);
            //If the validation error is decreasing with training, update the model parameters

            if(0 == global_config.s_method.compare("is")){
                if(0 == global_config.train_stop.compare("mse")){    //change this for masked-net
                    loss_to_compare = w_val_mse;
                    //loss_to_compare = val_re;
                }
                else if(0 == global_config.train_stop.compare("fn")) {
                    loss_to_compare = false_negatives;
                }
                else{
                    loss_to_compare = w_val_l_t;
                    //loss_to_compare = val_re;
                }

                if (loss_to_compare < prev_val_mse) {   // w_val_mse < prev_val_mse
                    //torch::save(model, "../best_model.pt");
                    best_mse = loss_to_compare ;           //best_mse = w_val_mse;
                    model = w_model;
                    epoch_t = epoch;
                    //best_l2_reg = l2_reg.template item<float>();
                    printf("Best model updated at epoch : %d",epoch);
                    mse_to_compare = loss_to_compare;
                    prev_val_mse=mse_to_compare;
                    count=0;
                }
                else  // previously val_mse>=prev_val_mse
                {
                    count++;
                    //if(0==global_config.do_100) {
                        if (count > global_config.stop_iter)
                            break;
                    //}
                }
            }
            else{
                //loss_to_compare = val_re;
                loss_to_compare = val_mse;
                if (loss_to_compare  < prev_val_mse) {
                    //torch::save(model, "../best_model.pt");
                    best_mse = val_mse;
                    model = w_model;
                    epoch_t = epoch;
                    printf("Best model updated at epoch : %d",epoch);
                    mse_to_compare = val_mse;
                    prev_val_mse=mse_to_compare;
                    prev_val_mse=loss_to_compare;
                    count=0;
                }
                else  // previously val_mse>=prev_val_mse
                {
                    count++;
                    //if(0==global_config.do_100) {
                        if (count > global_config.stop_iter)
                            break;
                    //}
                }
            }
            printf("Epoch number : %d, train mse : %f ", epoch, mse);

            /*
            if(to_save)
            {
                to_write_b.open(o_file_b,std::ios_base::app);

                if (to_write_b.is_open())
                {
                    printf("writing to the file %s", o_file_b.c_str());
                    if (isNet)
                        to_write_b << mse << '\t' << w_mse << '\t' << (float) val_mse << '\t' << (float)  w_val_mse << '\n' ;

                    to_write_b.close();
                }
            }
             */
        }

        /*
        if(to_save){
            to_write_b.open(o_file_b,std::ios_base::app);
            if (to_write_b.is_open())
            {
                printf("writing to the file %s", o_file_b.c_str());
                if (isNet)
                    to_write_b << epoch_t << '\n';
                to_write_b.close();
            }
        }*/

        if(to_save){
            to_write_b.open(o_file_b,std::ios_base::app);
            if (to_write_b.is_open())
            {
                //printf("writing to the file %s", o_file_b.c_str());
                if (isNet)
                    to_write_b <<  best_mse << '\t' ;
                to_write_b.close();
            }
        }

        printf("Out of training --epoch num %d", epoch_t);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        double duration = microseconds*1.0 / 1000000;
        duration /= n_epochs;
        this->WS()->time_Train += duration;
        this->WS()->count_Train++;

        std::string o_file = global_config.out_file + "plot.txt";
        std::ofstream to_write;
        to_write.open(o_file,std::ios_base::app);

        if (to_write.is_open())
        {
            printf("writing to the file %s", o_file.c_str());
            if (!isNet)
                to_write << bucket_num <<'\t' <<  global_config.width_problem << '\t'<< _nArgs << '\t' << epoch_t << '\t' << duration/3600 << '\t'<< global_config.loss_weight_mse << '\t' << global_config.loss_weight << '\t' << mse << '\t' << w_mse << '\t' << c_loss << '\t' << (float) val_mse << '\t' << (float)  w_val_mse << '\t' << val_c_loss<< '\t' ;
            else
                to_write << train_samples << '\t' << effective_N << '\t'<< bucket_num <<'\t' << global_config.width_problem << '\t'<< _nArgs << '\t' << epoch_t << '\t' << duration/3600 << '\t'  << mse << '\t' << w_mse << '\t' << val_mse << '\t' << w_val_mse << '\t' << float(train_re) << '\t' << val_re << '\t' ;
            to_write.close();
        }
        global_config.avg_val_mse = (global_config.avg_val_mse*(this->WS()->count_Train - 1) + val_mse)/ this->WS()->count_Train;
        global_config.avg_val_w_mse = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_val_mse)/ this->WS()->count_Train;
        global_config.avg_samples_req = (global_config.avg_samples_req*(this->WS()->count_Train - 1) + train_samples)/ this->WS()->count_Train;

#endif // INCLUDE_TORCH

        // write avg test_Error, val_error,
        printf("Out of training --");
    }

    void test(DATA_SAMPLES *DS_test, bool to_save)
	{
#ifdef INCLUDE_TORCH

	    //std::cout<<"in test -----";
	    if(model==NULL)
	        printf("MODEL IS NOT TRAINED!!");

        bool isNet=false;
        if (0 == global_config.network.compare("net")) {
            isNet=true;
        }

        size_t batch_idx = 0;
        float mse = 0.,w_mse=0.,c_loss=0.0, non_zeros=0, test_non_zeros=0, test_re =0, avg_e=0,max_e=-1,min_e=1000, w_avg_e=0,w_max_e=-1,w_min_e=1000,var_e;

        auto dataset = DS_test->map(torch::data::transforms::Stack<>());
        int64_t batch_size = global_config.batch_size;
        torch::DeviceType device_type;

        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device_type =  torch::kCUDA;//kCUDA
        } else {
            std::cout << "Training on CPU." << std::endl;
            device_type = torch::kCPU;
        }

        torch::Device device(device_type);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset,batch_size);
       // printf("Data loader --");
        int32_t c=0;
        int total_batch =0, neg_batch =0, neg=0;

        int true_positives=0, false_positives=0, true_negatives=0, false_negatives=0;
        int true_positives_batch=0, false_positives_batch=0, true_negatives_batch=0, false_negatives_batch=0;

        float correct=0,zero_one_loss=0;
        test_re=0;

        for (auto &batch : *data_loader) {
            c++;
            torch::Tensor loss,w_loss, loss_n, ln_out,ln_target,w, rel_error,avg_error,max_error,min_error,w_avg_error,w_max_error,w_min_error,var_error;

            auto imgs = batch.data;
            auto labels = batch.target.squeeze();
            imgs = imgs.to(device);
            labels = labels.to(device);
            auto output = model->forward(imgs,true,isNet);

            if (global_config.do_sum==0){
                if (global_config.do_weight==0)
                    w = test_samples*labels/sum_ln_abs;
                else
                    w = labels/sum_ln_abs;
            }
            else if (global_config.do_sum==2)
                w = exp(labels);
            else if (global_config.do_sum==1){
                if (global_config.do_weight==0){
                    w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln_abs*ln_max_value))*test_samples;
                }
                else{
                    //w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln_abs*ln_max_value));
                    w = exp(ln_min_value + labels*(ln_max_value - ln_min_value) -ln_sum)*test_samples;
                    w = exp((ln_min_value + labels*(ln_max_value - ln_min_value)) - ln_sum)*test_samples;
                }
            }
            else if (global_config.do_sum==-3){
                if (global_config.do_weight==0) {
                    w = test_samples*((ln_min_value + (labels+1) * (ln_max_value - ln_min_value)) / (2*(sum_ln_abs * ln_max_value)));
                }
                else
                    w = ((ln_min_value + (labels+1) * (ln_max_value - ln_min_value)) / (2*(sum_ln_abs * ln_max_value)));
            }
            else
                w = labels;

            if (isNet){
                if(0==global_config.loss.compare("wmse")){
                    //w_loss = (w*((output.x - labels).pow(2))).mean();
                    //loss = torch::nn::functional::mse_loss(labels, output.x);
                    if (global_config.l2_loss==1)
                        w_loss = (w*((output.x - labels).pow(2))).mean();
                    else
                        w_loss = (w*torch::abs((output.x - labels))).mean();
                       // w_loss = (ln_max_value-ln_min_value)*(w*torch::abs((output.x - labels))).mean();

                    if (global_config.l2_loss==1)
                        loss = torch::nn::functional::mse_loss(labels, output.x);
                    else
                        loss = torch::abs((output.x - labels)).mean();
                        //loss = (ln_max_value-ln_min_value)*torch::abs((output.x - labels)).mean();

                    rel_error = (torch::abs((output.x - labels)/labels)).mean();

                    if (global_config.l2_loss==0){
                        w_avg_error = (w*torch::abs(output.x - labels)).mean();
                        w_max_error = (w*torch::abs(output.x - labels)).max();
                        w_min_error = (w*torch::abs(output.x - labels)).min();
                        avg_error = torch::abs(output.x - labels).mean();
                        max_error = torch::abs(output.x - labels).max();
                        min_error = torch::abs(output.x - labels).min();
                        var_error = torch::abs(output.x - labels).var();
                    }
                    else{
                        w_avg_error = (w*torch::abs(output.x - labels)).mean();
                        w_max_error = (w*torch::abs(output.x - labels)).max();
                        w_min_error = (w*torch::abs(output.x - labels)).min();
                        avg_error = torch::abs(output.x - labels).mean();
                        max_error = torch::abs(output.x - labels).max();
                        min_error = torch::abs(output.x - labels).min();
                        var_error = torch::abs(output.x - labels).var();
                    }

                    auto label_binary_batch = torch::where(labels >= 0.5, torch::ones_like(labels), torch::zeros_like(labels));
                    auto zero_one_batch = torch::floor(output.x+0.5);
                    correct += (zero_one_batch == label_binary_batch).sum().template item<int>();

                    //zero_one_loss_batch  = torch::binary_cross_entropy(zero_one_batch, label_binary_batch);

                    /*
                    if(0==global_config.s_method.compare("is")){
                        w_avg_error = (w*torch::abs(output.x - labels)).mean();
                        w_max_error = (w*torch::abs(output.x - labels)).max();
                        w_min_error = (w*torch::abs(output.x - labels)).min();
                    }
                    else{
                        rel_error = (torch::abs((output.x - labels)/labels)).mean();
                        avg_error = torch::abs(output.x - labels).mean();
                        max_error = torch::abs(output.x - labels).max();
                        min_error = torch::abs(output.x - labels).min();
                    }*/
                }
                else{
                    ln_target = log(w);
                    ln_out = log(output.x);
                    std::cout<< output.x << log(output.x);
                    w_loss = (w*(ln_target-ln_out)).mean();
                    loss = (ln_target - ln_out).mean();
                    //std::cout << w << ln_out << w_loss << loss;
                }
            }
            else{
                auto label_binary = torch::where(labels > 0, torch::ones_like(labels), torch::zeros_like(labels));
                loss_n = torch::binary_cross_entropy(output.masked, label_binary);
                c_loss += loss_n.template item<float>();
                non_zeros = (label_binary==1).sum().template item<int>();
                test_non_zeros += non_zeros;
                neg_batch = (label_binary==0).sum().template item<int>();
                neg = neg + neg_batch;
                confusion_matrix(output.masked, label_binary, true_positives_batch, false_positives_batch, true_negatives_batch, false_negatives_batch);
                true_positives += true_positives_batch;
                false_positives += false_positives_batch;
                true_negatives += true_negatives_batch;
                false_negatives += false_negatives_batch;
                w_loss = (w*((output.x - labels).pow(2))).mean();
                loss = torch::nn::functional::mse_loss(labels, output.x);

                rel_error = (torch::abs((output.x - labels)/labels)).mean();
                w_avg_error = (w*torch::abs(output.x - labels)).mean();
                w_max_error = (w*torch::abs(output.x - labels)).max();
                w_min_error = (w*torch::abs(output.x - labels)).min();
                avg_error = torch::abs(output.x - labels).mean();
                max_error = torch::abs(output.x - labels).max();
                min_error = torch::abs(output.x - labels).min();
                var_error = torch::abs(output.x - labels).var();
                auto label_binary_batch = torch::where(labels >= 0.5, torch::ones_like(labels), torch::zeros_like(labels));
                auto zero_one_batch = torch::floor(output.x+0.5);
                correct += (zero_one_batch == label_binary_batch).sum().template item<int>();
            }

            mse += loss.template item<float>();
            w_mse += w_loss.template item<float>();
            test_re += rel_error.template item<float>();
            avg_e  += avg_error.template item<float>();
            w_avg_e += w_avg_error.template item<float>();
            var_e += var_error.template item<float>();

            if (max_e < max_error.template item<float>())
                max_e = max_error.template item<float>();
            if (min_e > min_error.template item<float>())
                min_e = min_error.template item<float>();
            if (w_max_e < w_max_error.template item<float>())
                w_max_e = w_max_error.template item<float>();
            if (w_min_e > w_min_error.template item<float>())
                w_min_e = w_min_error.template item<float>();
            batch_idx++;
        }

        mse /= (float) batch_idx;
        w_mse /= (float) batch_idx;
        test_re /= (float) batch_idx;
        avg_e /= (float) batch_idx;
        w_avg_e /= (float) batch_idx;
        var_e /= (float) batch_idx;

        zero_one_loss= 1-float(correct/test_samples);
        //std::cout<<zero_one_loss;

        if(!isNet)
            c_loss /= (float) batch_idx;
        printf("Test mse : %f ", mse );

        printf("\t Test: confusion matrix %d %d %d %d \t ",true_positives, false_positives, true_negatives, false_negatives);

        global_config.avg_test_mse = (global_config.avg_test_mse*(this->WS()->count_Train - 1) + mse)/ this->WS()->count_Train;
        global_config.avg_test_w_mse = (global_config.avg_test_w_mse*(this->WS()->count_Train - 1) + w_mse)/ this->WS()->count_Train;

        std::string o_file = global_config.out_file + "plot.txt";
        std::ofstream to_write;
        to_write.open(o_file,std::ios_base::app);

        //std::string o_file_b = global_config.out_file + "bucket.txt";
        //std::ofstream to_write_b;

        std::string o_file_b = global_config.out_file + "loss.txt";
        std::ofstream to_write_b;
        /*
        if(to_save)
        {
            to_write_b.open(o_file_b,std::ios_base::app);

            if (to_write_b.is_open())
            {
                printf("writing to the file %s", o_file_b.c_str());
                if (isNet)
                    to_write_b << mse << '\t' << w_mse << '\n' ;

                to_write_b.close();
            }
        }
        */
        to_save=1;

        if(to_save){
            to_write_b.open(o_file_b,std::ios_base::app);
            if (to_write_b.is_open())
            {
                //printf("writing to the file %s", o_file_b.c_str());
                if (isNet)
                    to_write_b <<  avg_e << '\t' << (ln_max_value-ln_min_value) << '\n';
                to_write_b.close();
            }
        }

        local_min_e=min_e*(ln_max_value-ln_min_value);
        local_max_e=max_e*(ln_max_value-ln_min_value);
        local_avg_e=avg_e*(ln_max_value-ln_min_value);
        local_w_min_e=w_min_e*(ln_max_value-ln_min_value);
        local_w_max_e=w_max_e*(ln_max_value-ln_min_value);
        local_w_avg_e=w_avg_e*(ln_max_value-ln_min_value);
        local_var_e = var_e*(ln_max_value-ln_min_value);

        global_config.seq_no +=1;

        if (global_config.max_test_err < zero_one_loss){
            global_config.max_test_err = zero_one_loss;
            global_config.max_width = _nArgs;
            global_config.max_seq_no = global_config.seq_no;
        }

        if(0==global_config.s_method.compare("is")){
            if (global_config.max_log_err < w_avg_e*(ln_max_value-ln_min_value)){
                global_config.max_avglog_err = w_avg_e*(ln_max_value-ln_min_value);
                global_config.max_log_err = max_e*(ln_max_value-ln_min_value);
                global_config.max_log_width = _nArgs;
                global_config.max_log_seq = global_config.seq_no;
            }
            if (global_config.max_wmse < w_avg_e)
                global_config.max_wmse = w_avg_e;
        } else{
            if (global_config.max_log_err < avg_e*(ln_max_value-ln_min_value)) {
                global_config.max_avglog_err = avg_e * (ln_max_value - ln_min_value);
                global_config.max_log_err = max_e * (ln_max_value - ln_min_value);
                global_config.max_log_width = _nArgs;
                global_config.max_log_seq = global_config.seq_no;
            }
            if (global_config.max_mse < avg_e)
                global_config.max_mse = avg_e;
        }

        if (global_config.max_lambda_e < max_e*(ln_max_value-ln_min_value)){
            global_config.max_lambda_e = max_e*(ln_max_value-ln_min_value);
            global_config.width_lambda_e = _nArgs;
        }

        global_config.avg_w_test_mse = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_mse)/ this->WS()->count_Train;
        global_config.avg_w_test_err = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_avg_e)/ this->WS()->count_Train;
        global_config.avg_lambda_test_err = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_avg_e)/ this->WS()->count_Train;

        if (zero_one_loss<global_config.epsilon)
            global_config.count+=1;

        if (to_write.is_open())
        {
            if(isNet)
                to_write << mse << '\t' << w_mse << '\t' << test_re << '\t'  << min_e << '\t' << max_e << '\t' << avg_e << '\t' << zero_one_loss<< '\t' << (ln_max_value-ln_min_value) << '\t' ;
            else
                to_write << mse << '\t'  << w_mse << '\t' << c_loss << '\t' << true_positives << '\t' << false_positives << '\t' << true_negatives << '\t' << false_negatives << '\t' << max_e << '\t' << avg_e << '\t' << zero_one_loss<< '\t' << (ln_max_value-ln_min_value) << "\t" ;
            to_write.close();
        }
      //  std::cout<<"out of test func ---";
      //  printf("Out of test function --");
#endif // INCLUDE_TORCH
    }

    void log_sum_exp(std::vector<float> arr, int count)
	{
       int isNet = 0 == global_config.network.compare("net");
            if(count > 0 ){
                double maxVal = arr[0];
                double sum = 0;
               // for (int i = 1 ; i < count ; i++){
               //     if (arr[i] > maxVal){
               //         maxVal = arr[i];
               //     }
               // }
                if (isNet){
                    for (int i = 0; i < count ; i++){
                        ln_sum += exp(arr[i] - ln_max_value);
			sum_ln += arr[i] - ln_min_value;
                        //sum_ln += arr[i]-ln_min_value;
                        sum_ln_abs += arr[i]/ln_max_value;
                    }
                    ln_sum = log(ln_sum) + ln_max_value;
                }
                else{
                    for (int i = 0; i < count ; i++) {
                        ln_sum += arr[i] / ln_max_value;
                        sum_ln_abs += arr[i] / ln_max_value;
                    }
                    ln_sum = ln_sum*ln_max_value;
                }
               // return log(ln_sum) + ln_max_value;
            }
        }

    DATA_SAMPLES *samples_to_data(std::vector<std::vector<int32_t>> & samples_signiture, std::vector<float> & samples_values, int32_t input_size, int32_t sample_size, int & non_zeros)
    {
	   // std::cout<<"in samples_to_Data ----";
	    float zero = 0.1*pow(10,-50);

	    for(int32_t i=0; i<sample_size; i++){
	        if (samples_values[i]<0)
	            global_config.negative_samples+=1;
	        if (global_config.do_sum==0){
                //printf("  value----  %f, max_value --- %f \n ",samples_values[i], ln_max_value);
                samples_values[i] = (float)samples_values[i]/ln_max_value;
            }
	        else if (global_config.do_sum==1){
                //printf("  value----  %f sum_value -----  %f, max_value --- %f \n ",samples_values[i], ln_sum, ln_max_value);
                //samples_values[i] = exp((float)samples_values[i] - ln_sum);
                samples_values[i] = ((float)samples_values[i] - ln_min_value)/(ln_max_value - ln_min_value);
              //  printf(" %f  value-- \n",samples_values[i]);
	        }
            else if (global_config.do_sum==-1){
                //printf("  value----  %f sum_value -----  %f, max_value --- %f \n ",samples_values[i], ln_sum, ln_max_value);
                //samples_values[i] = exp((float)samples_values[i] - ln_sum);
                samples_values[i] = 2*((float)samples_values[i] - ln_min_value)/(ln_max_value - ln_min_value) - 1;
                //  printf(" %f  value-- \n",samples_values[i]);
            }
	        else if (global_config.do_sum==2){
                samples_values[i] = (float)samples_values[i] - ln_sum;
	        }
            else if (global_config.do_sum==3){
                float a = (float)samples_values[i]-ln_min_value;
                float b = a/(sum_ln);
                //std::cout<<a<<b;
                samples_values[i] = (a)/(sum_ln) ;
                std::cout<<samples_values[i];
                //samples_values[i] = ((float)samples_values[i])/(sum_ln);
            }

            if (samples_values[i]>=zero){
                non_zeros+=1;
            }
        }

        DATA_SAMPLES *DS;
        DS = new DATA_SAMPLES(samples_signiture, samples_values, input_size, sample_size);

        //std::cout<<"out samples_to_Data ----";
        return DS;
    }


    void load_trained_model()
	{
		//we need to test the data here
    }

	FunctionNN(void)
		: Function()
	{
		// TODO own stuff here...
        printf("The void constructor");
        _TableData = NULL;

	}

	FunctionNN(Workspace *WS, ARP *Problem, int32_t IDX)
	{
	    printf("I am in the Constructor here");
       Function(WS, Problem, IDX);
       _TableData = NULL;
	}

	virtual ~FunctionNN(void)
	{
		Destroy() ;
	}


} ;

/*
    torch::Tensor FunctionNN::weighted_mse_loss(auto input, auto target, auto weight) {
        torch::Tensor temp = weight*((input - target)^2);
        return temp.sum();
    }
*/

inline Function *FunctionNNConstructor(void)
{
	return new FunctionNN;
}

} // namespace ARE

#endif // FunctionNN_HXX_INCLUDED
