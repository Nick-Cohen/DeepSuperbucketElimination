#ifndef FunctionNN_HXX_INCLUDED
#define FunctionNN_HXX_INCLUDED

#include <climits>
#include <stdlib.h>
#include <string>
#include <vector>

#include "Utils/Mutex.h"
#include "Utils/Sort.hxx"
#include "Problem/Globals.hxx"
#include "Problem/Function.hxx"
#include "Problem/Workspace.hxx"
#include "Net.h"
#include "DATA_SAMPLES.h"
#include <torch/torch.h>
#include "Config.h"
#include <chrono>
#include "Problem/Problem.hxx"

namespace BucketElimination { class Bucket ; class MiniBucket ; }

namespace ARE
{

class ARP ;

class FunctionNN : public Function
{

private:
   Net* model = NULL;//NOTE this one was auto in all the turorials   //TODO ask KALEV if this one can be nargs
   //Masked_Net * model = NULL;
   DATA_SAMPLES * DS = NULL;
   Config config;
   Masked_Net * masked_model = NULL;
//   torch::Tensor empty_tensor = torch::empty(_nArgs);
//   torch::DeviceType device_type_inf = torch::kCPU;
//   torch::Device device_inf(device_type_inf);

public :
    float ln_max_value = std::numeric_limits<float>::min();
    float sum_ln = 0.0;
    float ln_min_value = std::numeric_limits<float>::max();

    //float local_max_e=0,local_avg_e=0;

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
        auto start = std::chrono::high_resolution_clock::now();
        bool exp_converted = true; //TODO change this
        bool isNet;
        isNet = 0 == global_config.network.compare("net");
        if(isNet) {
            exp_converted = false;
        }
        torch::DeviceType device_type = torch::kCPU;
        torch::Device device(device_type);

        if(isNet) {
            model->to(device);
            model->eval();
        }
        else{
            masked_model->to(device);
            masked_model->eval();
        }

        auto empty_tensor = torch::empty(_nArgs);
        float* myData = empty_tensor.data_ptr<float>();

        int32_t var_domain_size;

        //normalizing inputs 
        for (int i=0; i <_nArgs; i++) {     
            if(isNet)          
                *myData++ = (float) BEPathAssignment[_ArgumentsPermutationList[i]]; //(float)BEPathAssignment[_ArgumentsPermutationList[i]]   
            else {
                var_domain_size = _Problem->K(_Arguments[i]) -1;
                *myData++ = (float) 2*BEPathAssignment[_ArgumentsPermutationList[i]]/var_domain_size -1;
            }                
        }

        torch::Tensor input = empty_tensor.resize_(_nArgs).clone();
        input = input.to(device);

        double out_value;
        torch::Tensor output;
        Masked_RET output_masked;

        if(isNet) {
            output = model->forward(input);
            out_value = (double)output.item<double>();
            out_value = ln_min_value + out_value*(ln_max_value - ln_min_value);
        }
        else {
            output_masked = masked_model->forward(input, true);
            out_value = (double)output_masked.x.item<double>();
            out_value = ln_min_value + out_value*(ln_max_value - ln_min_value);
            out_value = log(out_value);
        }

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        double duration = microseconds*1.0 / 1000000;
//        printf(" Table Duration: %f\n", duration);
        this->WS()->time_TableEntryEx += duration;
        this->WS()->count_TableEntryEx++;
        return out_value;
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
        torch::DeviceType device_type = torch::kCPU;
        torch::Device device(device_type);
        model->to(device);
        model->eval();

    }

	void Train(DATA_SAMPLES *DS_train, DATA_SAMPLES *DS_val, DATA_SAMPLES *DS_test, int bucket_num)
	{
//        DS->print_data();
        auto start = std::chrono::high_resolution_clock::now();

        _TableData = NULL;
        _TableSize = 0;

        global_config.h_dim = int(_nArgs*global_config.var_dim);

        //Init. model
        if(model == NULL)
            model = new Net(_nArgs);
        Net* w_model = new Net(_nArgs);

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

        torch::optim::Adam optimizer(w_model->parameters(), torch::optim::AdamOptions(config.lr));

        //some variable declarations
        int64_t n_epochs = global_config.n_epochs;
        float best_mse = std::numeric_limits<float>::max();
        float mse=0, val_mse=0, prev_val_mse=std::numeric_limits<float>::max();
        float w_mse=0, w_val_mse=0;
        int count=0, epoch,epoch_t=0;
        float zero = 0.0;
        float effective_N = 0, loss_to_compare=0. ;

        //Start training
        for (int epoch = 1; epoch <= n_epochs; epoch++) {
            printf("epoch %d",epoch);
            //Reset variables to 0 at the start of every epoch
            size_t batch_idx = 0, val_batch_idx = 0;
            mse = 0., val_mse= 0., w_mse = 0., w_val_mse = 0.;

            for (auto &batch : *data_loader_train) {
                effective_N = 0;
                torch::Tensor loss, w_loss;
                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(device);
                labels = labels.to(device);
                optimizer.zero_grad();
                auto output = w_model->forward(imgs);
			    
                //Calculate loss
                auto w =  ((labels*(ln_max_value - ln_min_value)) / (sum_ln));
                w_loss = (w*((output - labels).pow(2))).mean();  
                loss = torch::nn::functional::mse_loss(labels, output);

                if(0 == global_config.s_method.compare("is"))
                    w_loss.backward();
                else
                    loss.backward();
                
                optimizer.step();

                //Calculate effective sample size
                effective_N += (torch::sum(w).square()/torch::sum(w.square())).template item<float>();
                mse += loss.template item<float>();
                w_mse += w_loss.template item<float>();

                batch_idx++;
            }
            mse /= (float) batch_idx;
            w_mse /= (float) batch_idx;
            //train_re /= (float) batch_idx;
            printf("Epoch number : %d, train mse : %f train wmse : %f", epoch, mse,w_mse );

            torch::Tensor val_loss, w_val_loss;

            //Test updated model on validation set
            for (auto &batch : *data_loader_val) {

                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(device);
                labels = labels.to(device);
                auto output = w_model->forward(imgs);

                //Calculate loss
                auto w =  ((labels*(ln_max_value - ln_min_value)) / (sum_ln)); 
                w_val_loss = (w*((output - labels).pow(2))).mean();
                val_loss = torch::nn::functional::mse_loss(labels, output);
                val_mse += val_loss.template item<float>();
                w_val_mse += w_val_loss.template item<float>();

                val_batch_idx++;
            }

            val_mse /= (float) val_batch_idx ;
            w_val_mse /= (float) val_batch_idx;
            //val_re /= (float) val_batch_idx;
            printf("Validation  error ------- %f %f \n",val_mse,w_val_mse);

            //Check for stop condition
            if(0 == global_config.s_method.compare("is"))
                loss_to_compare = w_val_mse;
            else
                 loss_to_compare = val_mse;

            if (loss_to_compare < prev_val_mse) {   
                //torch::save(model, "../best_model.pt");
                best_mse = loss_to_compare ;          
                model = w_model;
                epoch_t = epoch;
                prev_val_mse = loss_to_compare ;
                count=0;
            }
            else  
            {
                count++;
                if (count > global_config.stop_iter)
                    break;
            }
        }

        std::cout<<train_samples<<"\t"<<effective_N<<std::endl;

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        double duration = microseconds*1.0 / 1000000;
        duration /= n_epochs;
        this->WS()->time_Train += duration;
        this->WS()->count_Train++;

        std::string o_file = global_config.out_file + "bucket-training.txt";
        std::ofstream to_write;
        to_write.open(o_file,std::ios_base::app);

        if (to_write.is_open())
        {
            printf("writing to the file %s", o_file.c_str());
            to_write << train_samples << '\t' << effective_N << '\t'<< bucket_num <<'\t' << _nArgs << '\t' << epoch_t << '\t' << duration/3600 << '\t'  << mse << '\t' << w_mse << '\t' << val_mse << '\t' << w_val_mse << '\t' ;
            to_write.close();
        }

        global_config.avg_val_mse = (global_config.avg_val_mse*(this->WS()->count_Train - 1) + val_mse)/ this->WS()->count_Train;
        global_config.avg_val_w_mse = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_val_mse)/ this->WS()->count_Train;
        global_config.avg_samples_req = (global_config.avg_samples_req*(this->WS()->count_Train - 1) + train_samples)/ this->WS()->count_Train;

        printf("Out of training --");
        test(DS_test);
    }

    void test(DATA_SAMPLES *DS_test) {
        if (model == NULL)
            printf("MODEL IS NOT TRAINED!!");

        auto dataset = DS_test->map(torch::data::transforms::Stack<>());
        int64_t batch_size = global_config.batch_size;
        torch::DeviceType device_type;

        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device_type =  torch::kCUDA;//kCUDA
        } else {
            std::cout << "testing on CPU." << std::endl;
            device_type = torch::kCPU;
        }

        torch::Device device(device_type);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset,batch_size);
        
        float correct=0,zero_one_loss=0;
        torch::Tensor loss, w_loss, avg_error, max_error;
        size_t batch_idx = 0;
        float mse = 0.,w_mse=0., avg_e=0,max_e=-1;
        
        for (auto &batch : *data_loader) {
            auto imgs = batch.data;
            auto labels = batch.target.squeeze();
            imgs = imgs.to(device);
            labels = labels.to(device);
            auto output = model->forward(imgs);

            auto w =  ((labels*(ln_max_value - ln_min_value)) / (sum_ln)); 
                    
            w_loss = (w*((output - labels).pow(2))).mean();
            loss = torch::nn::functional::mse_loss(labels, output);
            
            avg_error = torch::abs(output - labels).mean();
            max_error = torch::abs(output - labels).max();

            auto label_binary_batch = torch::where(labels >= 0.5, torch::ones_like(labels), torch::zeros_like(labels));
            auto zero_one_batch = torch::floor(output+0.5);
            correct += (zero_one_batch == label_binary_batch).sum().template item<int>();

            mse += loss.template item<float>();
            w_mse += w_loss.template item<float>();
            avg_e  += avg_error.template item<float>();
            if (max_e < max_error.template item<float>())
                max_e = max_error.template item<float>();

            batch_idx++;
        }
        mse /= (float) batch_idx;
        w_mse /= (float) batch_idx;
        avg_e /= (float) batch_idx;

        zero_one_loss= 1-float(correct/test_samples);

        global_config.avg_test_mse = (global_config.avg_test_mse*(this->WS()->count_Train - 1) + mse)/ this->WS()->count_Train;
        global_config.avg_test_w_mse = (global_config.avg_test_w_mse*(this->WS()->count_Train - 1) + w_mse)/ this->WS()->count_Train;

        std::string o_file = global_config.out_file + "bucket-training.txt";
        std::ofstream to_write;
        to_write.open(o_file,std::ios_base::app);

        if (to_write.is_open())
        {
            to_write << mse << '\t'  << w_mse << '\t' << max_e << '\t' << avg_e << '\t' << zero_one_loss<< '\t' ;
            to_write.close();
        }

        std::cout<<"Test mse :" << mse;
    }

    void confusion_matrix(torch::Tensor prediction,torch::Tensor truth, int &true_positives_batch, int &false_positives_batch, int &true_negatives_batch, int &false_negatives_batch){
        torch::Tensor confusion_vector = prediction / truth;
        true_positives_batch = (confusion_vector==1).sum().template item<int>();
        false_positives_batch = (confusion_vector== float('inf')).sum().template item<int>();
        true_negatives_batch = isnan(confusion_vector).sum().template item<int>();
        false_negatives_batch = (confusion_vector==0).sum().template item<int>();

    }

    void Train_ped(DATA_SAMPLES *DS_train, DATA_SAMPLES *DS_val,DATA_SAMPLES *DS_test,int bucket_num)
    {
        auto start = std::chrono::high_resolution_clock::now();

        _TableData = NULL;
        _TableSize = 0;

        global_config.h_dim = int(_nArgs*global_config.var_dim);

        if (masked_model == NULL)
            masked_model = new Masked_Net(_nArgs);

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

        torch::optim::Adam optimizer(w_model->parameters(), torch::optim::AdamOptions(config.lr));

        int64_t n_epochs = global_config.n_epochs;
        float best_mse = std::numeric_limits<float>::max();
        float mse=0, val_mse=0, w_mse=0, w_val_mse=0 ,prev_val_mse=std::numeric_limits<float>::max();
        int count=0, epoch, epoch_t=0;
        int true_positives_batch=0, false_positives_batch=0, true_negatives_batch=0, false_negatives_batch=0;
        int true_positives_t=0, false_positives_t=0, true_negatives_t=0, false_negatives_t=0;

        float effective_N = 0, loss_to_compare=0. ;

        for (int epoch = 1; epoch <= n_epochs; epoch++) {
            size_t batch_idx = 0, val_batch_idx=0;
            effective_N = 0;

            mse = 0., val_mse = 0., w_mse = 0., w_val_mse = 0. ; // mean squared error
            true_positives_t=0, false_positives_t=0, true_negatives_t=0, false_negatives_t=0;

            //Batch Training
            for (auto &batch : *data_loader_train) {
                torch::Tensor loss, w_loss, loss_n;

                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(device);
                labels = labels.to(device);
                optimizer.zero_grad();

                
                auto w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln*ln_max_value))*train_samples;
                auto output = w_model->forward(imgs, false);
                auto label_binary = torch::where(labels>0, torch::ones_like(labels), torch::zeros_like(labels));

                //Calculate loss
                loss_n =  torch::binary_cross_entropy(output.masked,label_binary);
                loss = 0.5*torch::nn::functional::mse_loss(labels, output.x) + loss_n;
                w_loss = 0.5*(w*((output.x - labels).pow(2))).mean() + loss_n;

                confusion_matrix(output.masked, label_binary, true_positives_batch, false_positives_batch, true_negatives_batch, false_negatives_batch);
                true_positives_t += true_positives_batch;
                false_positives_t += false_positives_batch;
                true_negatives_t += true_negatives_batch;
                false_negatives_t += false_negatives_batch;

                if(0 == global_config.s_method.compare("is"))
                    w_loss.backward();
                else
                    loss.backward();

                optimizer.step();

                effective_N += (torch::sum(w).square()/torch::sum(w.square())).template item<float>();
                mse += loss.template item<float>();
                w_mse += w_loss.template item<float>();

                batch_idx++;
            }

            printf("\t  %d %d %d %d \t ",true_positives_t, false_positives_t, true_negatives_t, false_negatives_t);
            mse /= (float) batch_idx;
            w_mse /= (float) batch_idx;

            printf(" Mean squared error: %f wmse : %f\n", mse, w_mse);

            torch::Tensor val_loss, w_val_loss, val_loss_n;

            //Calculate loss on validation set
            for (auto &batch : *data_loader_val) {
                auto imgs = batch.data;
                auto labels = batch.target.squeeze();
                imgs = imgs.to(device);
                labels = labels.to(device);

                auto w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln*ln_max_value))*val_samples;

                auto output = w_model->forward(imgs, true);
                auto label_binary = torch::where(labels>0, torch::ones_like(labels), torch::zeros_like(labels));
                auto val_loss_n =  torch::binary_cross_entropy(output.masked,label_binary);

                val_loss = 0.5*torch::nn::functional::mse_loss(labels, output.x) + val_loss_n;
                w_val_loss = 0.5*(w*((output.x - labels).pow(2))).mean() + val_loss_n;

                val_mse += val_loss.template item<float>();
                w_val_mse += w_val_loss.template item<float>();

                val_batch_idx++;
            }

            val_mse /= (float) val_batch_idx ;
            w_val_mse /= (float) val_batch_idx ;

            //Stop condition 
            if(0 == global_config.s_method.compare("is")) 
                loss_to_compare = w_val_mse;
            else
                loss_to_compare = val_mse;

            if (loss_to_compare < prev_val_mse) {
                //torch::save(model, "../best_model.pt");
                std::cout<<"saving model now..."<< prev_val_mse << '\t ' << loss_to_compare ;
                best_mse = loss_to_compare;
                masked_model = w_model;
                epoch_t = epoch;
                prev_val_mse= loss_to_compare;
                count=0;
            }
            else  
            {
                std::cout<<"NOT saving model now..."<< prev_val_mse << '\t ' << loss_to_compare ;
                count++;
                if (count > global_config.stop_iter)
                    break;
            }

         //   printf("Epoch number : %d, train wmse : %f ", epoch, w_mse);
        }

        printf("Out of training --epoch num %d", epoch_t);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        long long microseconds = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        double duration = microseconds*1.0 / 1000000;
        duration /= n_epochs;
        this->WS()->time_Train += duration;
        this->WS()->count_Train++;

        std::string o_file = global_config.out_file + "bucket-training.txt";
        std::ofstream to_write;
        to_write.open(o_file,std::ios_base::app);

        if (to_write.is_open())
        {
            printf("\n writing to the file %s", o_file.c_str());
            to_write << train_samples << '\t' << effective_N << '\t' << bucket_num <<'\t' <<  _nArgs << '\t' << epoch_t << '\t' << duration/3600 << '\t' << mse << '\t' << w_mse << '\t' << (float) val_mse << '\t' << (float)  w_val_mse << '\t' ;
            to_write.close();
        }

        global_config.avg_val_mse = (global_config.avg_val_mse*(this->WS()->count_Train - 1) + val_mse)/ this->WS()->count_Train;
        global_config.avg_val_w_mse = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_val_mse)/ this->WS()->count_Train;
        global_config.avg_samples_req = (global_config.avg_samples_req*(this->WS()->count_Train - 1) + train_samples)/ this->WS()->count_Train;
        // write avg test_Error, val_error,
        printf("\n Out of training ---------------");
        ped_test(DS_test);
    }

        void Train_ped_before(DATA_SAMPLES *DS_train, DATA_SAMPLES *DS_val,DATA_SAMPLES *DS_test,int bucket_num)
        {
            auto start = std::chrono::high_resolution_clock::now();
            _TableData = NULL;
            _TableSize = 0;
            
            global_config.h_dim = int(_nArgs*global_config.var_dim);
            if (masked_model == NULL)
                masked_model = new Masked_Net(_nArgs);

            Masked_Net * w_model = new Masked_Net(_nArgs);
            torch::DeviceType device_type;
            if (torch::cuda::is_available()) {
                std::cout << "CUDA available! Training on GPU." << std::endl;
                device_type =  torch::kCUDA;//kCUDA
            } else {
                std::cout << "Training on CPU." << std::endl;
                device_type = torch::kCPU;
            }
            printf("hmm okay I am here");
            torch::Device device(device_type);
            w_model->to(device);
            w_model->train();

            auto dataset_train = DS_train->map(torch::data::transforms::Stack<>());
            int64_t batch_size = global_config.batch_size;
            auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset_train,batch_size);

            auto dataset_val = DS_val->map(torch::data::transforms::Stack<>());
            auto data_loader_val = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset_val,batch_size);

            //auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset,batch_size);
            torch::optim::Adam optimizer(w_model->parameters(), torch::optim::AdamOptions(config.lr));

            int64_t n_epochs = config.n_epochs;
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

            int t=0,b=0;

            float last_loss = 0.0;

            float effective_N = 0;
            for (int epoch = 1; epoch <= n_epochs; epoch++) {
                size_t batch_idx = 0, val_batch_idx=0;
                effective_N = 0;
                mse = 0.; // mean squared error
                val_mse = 0.;
                c_loss = 0.;
                val_c_loss = 0.;
                w_mse = 0.; // mean squared error
                w_val_mse = 0.;
                train_re=0.0;
                val_re=0.0;


                for (auto &batch : *data_loader_train) {
                    torch::Tensor loss, w_loss, loss_n;

                    auto imgs = batch.data;
                    auto labels = batch.target.squeeze();
                    imgs = imgs.to(device);
                    labels = labels.to(device);
                    optimizer.zero_grad();

                    w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln*ln_max_value))*train_samples;
                        
                    //std::cout<<w<<"\n";

                    //std::cout<<"weights printed ----";
                    auto output = w_model->forward(imgs, false);
                    auto label_binary = torch::where(labels>0, torch::ones_like(labels), torch::zeros_like(labels));
                    loss_n =  torch::binary_cross_entropy(output.masked,label_binary);


                    loss = 0.5*torch::nn::functional::mse_loss(labels, output.x) + loss_n;
                    w_loss = 0.5*(w*((output.x - labels).pow(2))).mean() + loss_n;

                    confusion_matrix(output.masked, label_binary, true_positives_batch, false_positives_batch, true_negatives_batch, false_negatives_batch);
                    true_positives_t += true_positives_batch;
                    false_positives_t += false_positives_batch;
                    true_negatives_t += true_negatives_batch;
                    false_negatives_t += false_negatives_batch;

                    if(0 == global_config.s_method.compare("is")){
                        w_loss.backward();
                    }
                    else{
                        loss.backward();
                    }

                    optimizer.step();
                    //effective_N += 1/(torch::sum(w.square())).template item<float>();
                    effective_N += (torch::sum(w).square()/torch::sum(w.square())).template item<float>();
                    mse += loss.template item<float>();
                    float l = loss.template item<float>();
                    float w_l = w_loss.template item<float>();

                    mse += l;
                    w_mse += w_l;
                    //train_re += a;

                    batch_idx++;
                }

               // count++;
                printf("\t  %d %d %d %d \t ",true_positives_t, false_positives_t, true_negatives_t, false_negatives_t);
                mse /= (float) batch_idx;
                w_mse /= (float) batch_idx;
                //printf("the total epochs of this training is finished\n");
                //mse /= (float) batch_idx;
                printf(" Mean squared error: %f wmse : %f\n", mse, w_mse);

                batch_idx = 0;
                torch::Tensor val_loss, w_val_loss, val_loss_n;
                true_positives=0, false_positives=0, true_negatives=0, false_negatives=0;
                for (auto &batch : *data_loader_val) {
                    auto imgs = batch.data;
                    auto labels = batch.target.squeeze();
                    imgs = imgs.to(device);
                    labels = labels.to(device);

                    
                    w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln*ln_max_value))*val_samples;
                        

                    auto output = w_model->forward(imgs, true);
                    auto label_binary = torch::where(labels>0, torch::ones_like(labels), torch::zeros_like(labels));
                    auto val_loss_n =  torch::binary_cross_entropy(output.masked,label_binary);
                    //printf("labels shape %d, out shape %d", labels.sizes(),output.x.sizes());

                    val_loss = 0.5*torch::nn::functional::mse_loss(labels, output.x) + val_loss_n;
                    w_val_loss = 0.5*(w*((output.x - labels).pow(2))).mean() + val_loss_n;
                   // printf("loss_n shape %d", val_loss_n.sizes());
                    val_mse += val_loss.template item<float>();
                    w_val_mse += w_val_loss.template item<float>();
                    //HEREEEE ************************************s*******************
                    batch_idx++;
                    val_batch_idx++;
                }

                val_mse /= (float) val_batch_idx ;
                w_val_mse /= (float) val_batch_idx ;
                if(0 == global_config.s_method.compare("is")) {
                    loss_to_compare = w_val_mse;
                }
                else
                    loss_to_compare = val_mse;

                if (loss_to_compare < prev_val_mse) {
                    //torch::save(model, "../best_model.pt");
                    best_mse = loss_to_compare;
                    masked_model = w_model;
                    epoch_t = epoch;
                    printf("I AM HERE --------------- Best model updated at epoch : %d",epoch);
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

                printf("Epoch number : %d, train mse : %f ", epoch, mse);
            }

            printf("Out of training --epoch num %d", epoch_t);
//            auto stop = std::chrono::high_resolution_clock::now();
//            auto duration = duration_cast <std::chrono::microseconds> (stop - start);
//            std::cout << "Time taken for training here: "<< duration.count() << " microseconds" << std::endl;
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
                printf("\n writing to the file %s", o_file.c_str());
                to_write << train_samples << '\t' << effective_N << '\t' << bucket_num <<'\t' <<  global_config.width_problem << '\t'<< _nArgs << '\t' << epoch_t << '\t' << duration/3600 << '\t' << mse << '\t' << w_mse << '\t' << c_loss << '\t' << (float) val_mse << '\t' << (float)  w_val_mse << '\t' << val_c_loss<< '\t' ;
                to_write.close();
            }

            global_config.avg_val_mse = (global_config.avg_val_mse*(this->WS()->count_Train - 1) + val_mse)/ this->WS()->count_Train;
            global_config.avg_val_w_mse = (global_config.avg_val_w_mse*(this->WS()->count_Train - 1) + w_val_mse)/ this->WS()->count_Train;
            global_config.avg_samples_req = (global_config.avg_samples_req*(this->WS()->count_Train - 1) + train_samples)/ this->WS()->count_Train;
            // write avg test_Error, val_error,
            printf("\n Out of training ---------------");

            ped_test(DS_test);
        }





    void ped_test(DATA_SAMPLES *DS_test) {
        //std::cout<<"in test -----";
        if (masked_model == NULL)
            printf("MODEL IS NOT TRAINED!!");

        auto dataset = DS_test->map(torch::data::transforms::Stack<>());
        int64_t batch_size = global_config.batch_size;
        torch::DeviceType device_type;

        if (torch::cuda::is_available()) {
            std::cout << "CUDA available! Training on GPU." << std::endl;
            device_type =  torch::kCUDA;//kCUDA
        } else {
            std::cout << "testing on CPU." << std::endl;
            device_type = torch::kCPU;
        }

        torch::Device device(device_type);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset,batch_size);
        // printf("Data loader --");
        size_t batch_idx = 0;
        float mse = 0., w_mse=0., c_loss=0.0, non_zeros=0, test_non_zeros=0,  avg_e=0, max_e=-1 ;
        int total_batch =0, neg_batch =0, neg=0;
        int true_positives=0, false_positives=0, true_negatives=0, false_negatives=0;
        int true_positives_batch=0, false_positives_batch=0, true_negatives_batch=0, false_negatives_batch=0;
        float correct=0,zero_one_loss=0;

        for (auto &batch : *data_loader) {
            torch::Tensor loss,w_loss, loss_n, ln_out,avg_error,max_error;

            auto imgs = batch.data;
            auto labels = batch.target.squeeze();
            imgs = imgs.to(device);
            labels = labels.to(device);
            auto output = masked_model->forward(imgs,true);

            
            auto w = ((ln_min_value + labels*(ln_max_value - ln_min_value))/(sum_ln*ln_max_value))*test_samples;
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

            w_loss = (w*((output.x - labels).pow(2))).mean() + loss_n;
            loss = torch::nn::functional::mse_loss(labels, output.x) + loss_n;
            
            auto label_binary_batch = torch::where(labels >= 0.5, torch::ones_like(labels), torch::zeros_like(labels));
            auto zero_one_batch = torch::floor(output.x+0.5);
            correct += (zero_one_batch == label_binary_batch).sum().template item<int>();

            mse += loss.template item<float>();
            w_mse += w_loss.template item<float>();
            
            batch_idx++;
        }
        mse /= (float) batch_idx;
        w_mse /= (float) batch_idx;

        zero_one_loss= 1-float(correct/test_samples);

        global_config.avg_test_mse = (global_config.avg_test_mse*(this->WS()->count_Train - 1) + mse)/ this->WS()->count_Train;
        global_config.avg_test_w_mse = (global_config.avg_test_w_mse*(this->WS()->count_Train - 1) + w_mse)/ this->WS()->count_Train;

        std::string o_file = global_config.out_file + "bucket-training.txt";
        std::ofstream to_write;
        to_write.open(o_file,std::ios_base::app);

        if (to_write.is_open())
        {
            to_write << mse << '\t'  << w_mse << '\t' << true_positives << '\t' << false_positives << '\t' << true_negatives << '\t' << false_negatives << '\t' ;
            to_write.close();
        }
        std::cout<<"Test mse :" << mse;
    }

    void log_sum_exp(std::vector<float> arr, int count){
        int isNet = 0 == global_config.network.compare("net");

        if(count > 0 ){
            double maxVal = arr[0];
            double sum = 0;
            for (int i = 1 ; i < count ; i++){
                 if (arr[i] > ln_max_value)
                     ln_max_value = arr[i];
        	     if (arr[i]<ln_min_value)
        		     ln_min_value = arr[i];
             }

            std::cout<<"max and min value --" << ln_max_value << '\t' <<ln_min_value;
            if (isNet){
                for (int i = 0; i < count ; i++)
		            sum_ln += arr[i] - ln_min_value;

            }
            else {
                for (int i = 0; i < count; i++) 
                    sum_ln += arr[i] / ln_max_value;
            }

        }

    }


    DATA_SAMPLES * samples_to_data(std::vector<std::vector<float> > samples_signiture,std::vector<float> samples_values, int32_t input_size, int32_t sample_size){
        
        // Normalize each sample value in [0,1]
        for(int32_t i=0; i<sample_size; i++)
        {
           // std::cout<<samples_values[i];
            samples_values[i] = ((float)samples_values[i] - ln_min_value)/(ln_max_value - ln_min_value);
          // std::cout<<samples_values[i];
           //exit(0);
        }

        std::cout<<samples_values[10];
        DATA_SAMPLES *DS;
        DS = new DATA_SAMPLES(samples_signiture, samples_values, input_size, sample_size);
        return DS;
    }

    torch::Tensor get_masked_loss(torch::Tensor mask, torch::Tensor label){
	    auto label_binary = torch::where(label>0, torch::ones_like(label), torch::zeros_like(label));
        torch::Tensor loss_here = torch::nll_loss(torch::log_softmax(mask,1), label_binary);
	    return loss_here;
	}

    void load_trained_model(){

    //we need to test the data here
    }

	FunctionNN(void)
		:
		Function()
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

inline FunctionNN * FunctionNNConstructor(void)
{
//    myData = empty_tensor.data_ptr<float>();
	return new FunctionNN;
}

} // namespace ARE

#endif // FunctionNN_HXX_INCLUDED
