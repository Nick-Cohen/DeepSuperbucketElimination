//
// Created by yasaman razeghi
//

#include <iostream>

//#define INCLUDE_TORCH

//this is the model architectuer for grid-> the simplest version
#ifndef BESAMPLING_NET_H
#define BESAMPLING_NET_H

#include <NNConfig.h>

#ifdef INCLUDE_TORCH

#include <torch/torch.h>
#include <torch/nn/module.h>

struct Net : torch::nn::Module {
    int64_t h_dim = global_config.h_dim;
    Net(int64_t Input_size)
            : linear1(torch::nn::Linear(Input_size,  h_dim)),
              linear2(torch::nn::Linear( h_dim, h_dim)),
              linear3(torch::nn::Linear( h_dim, 1)),
              linear4(torch::nn::Linear( h_dim, 1))
    {
        register_module("linear1", linear1);
        register_module("linear2", linear2);
        register_module("linear3", linear3);
        register_module("linear4", linear4);
    }
    torch::Tensor forward(torch::Tensor input) {
        torch::Tensor x = torch::relu(linear1(input));
//        x =  torch::relu(linear2(x));
        x = linear2(x);
       // x = torch::relu(linear1(input));
       // x = torch::relu(linear2(x)); //sigmoid?
        x = linear3(x);

        return x.view(-1);
    }
    torch::nn::Linear linear1, linear2, linear3, linear4;
};
//TORCH_MODULE(Net);

struct Masked_RET {
    torch::Tensor x;
    torch::Tensor masked;
};

struct Masked_Net : torch::nn::Module {
    int64_t h_dim = global_config.h_dim;
    Masked_Net(int64_t Input_size)
            : linear1(torch::nn::Linear(Input_size, h_dim)),
              linear2(torch::nn::Linear(h_dim, h_dim)),
              linear3(torch::nn::Linear(h_dim, 1)),
              linear4(torch::nn::Linear(h_dim, 1))
    {
        register_module("linear1", linear1);
        register_module("linear2", linear2);
        register_module("linear3", linear3);
        register_module("linear4", linear4);
    }
    Masked_RET forward(torch::Tensor input, bool isTest) {
        Masked_RET ret_Masked_Net;
        torch::Tensor x = torch::relu(linear1(input));
        torch::Tensor mask = torch::sigmoid(linear4(x));
        x = torch::relu(linear2(x));
        x = torch::nn::functional::softplus(linear3(x));
//        x =  torch::relu(linear2(x));
        if (isTest == true){
            mask = torch::floor(mask+0.5);
            ret_Masked_Net.x = x*mask.view(-1);
            ret_Masked_Net.masked = mask.view(-1);
            return ret_Masked_Net;
        }
        ret_Masked_Net.x = x.view(-1);
        ret_Masked_Net.masked = mask.view(-1);
        return ret_Masked_Net;
    }
    torch::nn::Linear linear1, linear2, linear3, linear4;
};

#endif INCLUDE_TORCH

#endif //BESAMPLING_NET_H
