#%%
from NN_Train import Net, NN_Data
import torch as t
import torch.nn as nn

full_data = NN_Data('/home/cohenn1/SDBE/GAN_tests/bucket-output-fn21.xml', device='cuda')
domain_sizes = full_data.domain_sizes
input_len = len(full_data.input_vectors[0])

# Initialize GAN to generate uniformly random assignments to  input variables
# Generate n samples
# Train predictor on samples
# Backprop -loss wrt Gumbel Softmaxed samples (maybe just use test set?)

class Adversary(nn.module):
    
    def __init__(self, predictor, latent_size=100, num_iter=10, device='cuda'):
        super(Adversary, self).__init__()
        self.predictor = predictor
        self.latent_size = latent_size
        self.num_iter = num_iter
        self.device = device
        predictor_input_len = len(predictor.data.input_vectors[0])
        self.fc1 = nn.Linear(latent_size, 100, device=self.device)
        self.fc2 = nn.Linear(100, predictor_input_len)a
        self.activation_function = nn.Softplus().to(self.device)
        
    def forward(self, x):
        out = 