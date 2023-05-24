//
// Created by yasaman razeghi
//

#ifndef BESAMPLING_DATA_SAMPLES_H
#define BESAMPLING_DATA_SAMPLES_H
#include <torch/torch.h>
#include <torch/csrc/api/include/torch/data/datasets/base.h>

class DATA_SAMPLES : public torch::data::Dataset<DATA_SAMPLES> {

private:
    torch::Tensor states_, labels_;
    int data_size = 0;
    int input_size = 0;
public:
    explicit DATA_SAMPLES() {
        printf("helo I am here");
    };

    torch::data::Example<> get(size_t index) override{
        // You may for example also read in a .csv file that stores locations
        // to your data and then read in the data at this step. Be creative.
        return {states_[index], labels_[index]};
    };
    torch::optional<size_t> size() const override {
        return data_size;
    };

    DATA_SAMPLES(std::vector<std::vector<float> > samples_signiture, std::vector<float> samples_values, int32_t input_size, int32_t sample_size){
        data_size = (int)sample_size;
        this->input_size = (int)input_size;
        printf("Happy to have this haahahahaah %f \n", samples_values[0]);

        auto empty_tensor = torch::empty(sample_size*input_size);
        float* myData = empty_tensor.data_ptr<float>();
        for(int i=0; i<sample_size; i++)
            for(int j =0; j<input_size; j++)
                *myData++ = (float)samples_signiture[i][j];
        states_ = empty_tensor.resize_({sample_size,input_size}).clone();

        auto empty_tensor2 = torch::empty(sample_size);
        float* myDatal = empty_tensor2.data_ptr<float>();
        for(int i=0; i<sample_size; i++)
            *myDatal++ = (float)samples_values[i];
        labels_ = empty_tensor2.resize_({sample_size}).clone();

    };

    void print_data(){
        printf("this dataset is the size %d \n", data_size);
        for (int i=0; i<data_size; i++){
//            for (int j=0; j<input_size; j++)
//                std::cout<<states_[i][j];
            std::cout<<"the label is"<<labels_[i]<<std::endl;
        }
        printf("end of data");
    };
};

#endif //BESAMPLING_DATA_SAMPLES_H
