
#ifndef BESAMPLING_CONFIG_H
#define BESAMPLING_CONFIG_H

#include <string>

//Here I create this struct to set the parameters:
typedef struct Config{
    float lr = 0.001;
    int32_t n_epochs = 500; //was 10
    std::string out_file, out_file2;
    float avg_val_mse=0.0,avg_test_mse=0.0, avg_val_w_mse=0.0,avg_test_w_mse=0.0, avg_samples_req = 0 ;

    std::string network="masked_net";
    std::string s_method="is";
    int width_problem = 20;
    int n_layers=2;
    int32_t batch_size = 256;   
    int stop_iter=2;
    double epsilon=0.25;
    int var_dim=1;
    int h_dim=0;
    //int input_norm=0;

}Config_NN;

extern Config_NN global_config;

#endif //BESAMPLING_CONFIG_H
