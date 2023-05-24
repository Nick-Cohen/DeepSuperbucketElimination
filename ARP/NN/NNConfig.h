#ifndef BESAMPLING_CONFIG_H
#define BESAMPLING_CONFIG_H

#include <stdint.h>
#include <string>

//Here I create this struct to set the parameters:

typedef struct Config{
    int32_t sample_size = 500000;
    float lr = 0.001;
    float e = 1e-8;
    //float weight_decay = 0.00001;
    float l2_lambda= 0.000001;
    float r =0;
    int32_t n_epochs = 500; //was 10
    int32_t dev_size = 10000;
    std::string out_file, out_file2;

    float avg_val_mse=0.0,avg_test_mse=0.0, avg_val_w_mse=0.0,avg_test_w_mse=0.0 ;
    std::string network="masked_net";
    std::string s_method="is";
    int width_problem = 20;
    float loss_weight = 1;
    float loss_weight_mse =1;
    int num_hidden = 5;
    float dropout = 0.5;
    int32_t h_dim = 100;
    int32_t batch_size = 256;   //1024
    std::string train_stop="mse";
    std::string loss="wmse";
    int do_sum=1;
    int do_weight=0;
    int stop_iter=2;
    double epsilon=0.25;
   // bool do_100=0;s
    int l2=1;
    int n_layers=2;
    int var_dim=1;
    int adam_opt=1;
    int l2_loss=1;
    int reg=0;
    int var_samples=1;
    int vareps=0;
    int alpha=200;
    int prev_e=0;
    int c=4;
    float max_test_err = 0;
    float max_width = 0;
    int count=0;
    int seq_no = 0;
    int max_seq_no=0;
    float avg_samples_req = 0;
    float avg_w_test_err=0;
    float max_log_err = 0;
    float max_avglog_err=0;
    float max_wmse=0;
    float max_lambda_e=0;
    float width_lambda_e =0;
    float max_mse=0;
    int max_log_width=0;
    int max_log_seq=0;
    float avg_lambda_test_err=0;
    float avg_w_test_mse=0;
    float negative_samples=0;
    int lb=0;
    int up=0;
    int input_norm=0;

} Config_NN ;

extern Config_NN global_config ;

#endif //BESAMPLING_CONFIG_H
