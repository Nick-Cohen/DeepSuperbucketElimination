#include <torch/torch.h>
#include <vector>

torch::Tensor one_hot_encode(torch::Tensor input, int num_classes, bool lower_dim) {
    int num_samples = input.size(0);
    int num_vars = lower_dim ? num_classes - 1 : num_classes;
    
    // Initialize output tensor with zeros
    torch::Tensor output = torch::zeros({num_samples, num_vars});

    // Iterate over each element in the input tensor
    for (int i = 0; i < num_samples; i++) {
        int value = input[i].item<int>();

        // Check if value is in the valid range
        if (value >= 0 && value < num_classes) {
            // If lower_dim is true, ignore the first category
            if (lower_dim && value > 0) {
                output[i][value - 1] = 1;
            } else if (!lower_dim) {
                output[i][value] = 1;
            }
        }
    }

    return output;
}
int main() {
    torch::Tensor input = torch::tensor({0, 1, 2, 1});
    torch::Tensor encoded = one_hot_encode(input, 3, true);
    std::cout << encoded << std::endl;
    return 0;
}
