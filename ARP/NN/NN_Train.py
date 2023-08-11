# To run: python NN-Train.py --samples samplesFileName.xml --nn_path pathToSavedNN --done_path pathToFileIndicatingTrainingIsDone
# nn_path is 'nn-samples-varIndex1;varIndex2;... .pt' by default
# done_path is 'training-complete-varIndex1;varIndex2;... .pt' by default



# Load samples and domain size from file, also potentially is_log_space is_masked
# Convert samples to one-hot
# Train NN
# Write NN weights to file

#%%
import torch as t
import torch.nn.functional as F
import torch.nn as nn
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Tuple, IO
import argparse

class NN_Data:

    def __init__(self, file_name = None, processed_samples = None, values = None, device = 'cpu'):
        self.file_name = file_name
        self.num_samples: int
        self.features_per_sample: int
        self.is_log_space: bool
        self.max: float
        self.min: float
        self.sum: float # log sum of non-logspace values
        self.signatures: t.Tensor
        self.input_vectors: t.Tensor
        self.values: t.Tensor
        self.is_positive: t.BoolTensor # (num_samples,1) indicates if table has non-zero value: True or zero: False
        self.domain_sizes: t.IntTensor
        self.device = device
        if file_name is not None:
            try:
                 self.parse_samples()
            except IOError:
                print(f"Error reading file %s", self.file_name)


    def parse_samples(self):
        tree = ET.parse(self.file_name)
        root = tree.getroot()

        signatures_list = []
        values_list = []
        num_samples = 0
        for sample in root.iter('sample'):
            num_samples += 1
            signature = sample.get('signature')
            # Convert signature string '0;1;0;1;...' to a list of integers [0, 1, 0, 1, ...]
            signature = list(map(int, signature.split(';')))
            signatures_list.append(signature)

            value = float(sample.get('value'))
            values_list.append(value)
        self.num_samples = num_samples

        # Convert lists to PyTorch tensors
        signatures_tensor = t.tensor(signatures_list).to(self.device)
        values_tensor = t.tensor(values_list).to(self.device)

        # Replace -inf with a large negative number
        values_tensor[values_tensor == float('-inf')] = -1e10
        
        self.max_value = float(t.exp(t.log(t.tensor(10)) * max(values_tensor))) # take max value exponentiated out of log space

        self.signatures, self.values = signatures_tensor, values_tensor
        self.values[self.values == float('-inf')] = -1e10
        self.is_positive = self.values.ne(float('-inf'))
        

        # Get 'outputfnvariabledomainsizes' attribute and convert it to a list of integers
        domain_sizes = [int(x) for x in root.get('outputfnvariabledomainsizes').split(';')]
        self.domain_sizes = t.IntTensor(domain_sizes).to(self.device)
        self.input_vectors = (self.one_hot_encode(self.signatures).float()).to(self.device)



    def one_hot_encode(self, signatures: t.IntTensor, lower_dim = True):
        # transforms (num_samples, num_vars) tensor to (num_samples, sum(domain_sizes)) one hot encoding

        num_samples, num_vars = signatures.shape

        if lower_dim: # send n domain variables to n-1 vector
            one_hot_encoded_samples = t.cat([F.one_hot(signatures[:, i], num_classes=self.domain_sizes[i])[:, 1:] for i in range(num_vars)], dim=-1)
        else:
            one_hot_encoded_samples = t.cat([F.one_hot(signatures[:, i], num_classes=self.domain_sizes[i]) for i in range(num_vars)], dim=-1)

        return one_hot_encoded_samples
    
    

    

# %%
class Net(nn.Module):
    # def __init__(self, input_size, hidden_size, output_size):
    def __init__(self, nn_data: NN_Data, epochs = 1000):
        super(Net, self).__init__()
        self.nn_data = nn_data
        self.max_value = nn_data.max_value
        self.num_samples = self.nn_data.num_samples
        self.epochs = epochs
        self.device = nn_data.device
        self.values = nn_data.values.float().to(self.device)
        input_size, hidden_size = len(nn_data.input_vectors[0]), len(nn_data.input_vectors[0])*2
        output_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(self.device)
        self.relu = nn.ReLU().to(self.device)

        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.L1Loss()
        self.optimizer = t.optim.Adam(self.parameters())

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out
    
    def train_model(self, batch_size=100):
        epochs = self.epochs
        X=self.nn_data.input_vectors#.float()
        assert(X.device.type==self.device) ############################################################
        num_batches = int(self.num_samples / batch_size)
        print(self.num_samples, batch_size, num_batches)

        for epoch in range(epochs):
            for i in range(num_batches):
                # Get mini-batch
                X_batch = X[i*batch_size : (i+1)*batch_size]
                values_batch = self.values[i*batch_size : (i+1)*batch_size]
                assert(values_batch.device.type==self.device) #################################

                # Predict
                pred_values = self(X_batch)
                assert(pred_values.device.type==self.device) #################################


                # Convert predictions and values from log space to original scale
                pred_values_exp = t.exp(pred_values * t.log(t.tensor(10))) # I might need to change the order I do these exponentiations for numerical precision reasons
                values_exp = t.exp(values_batch.view(-1, 1) * t.log(t.tensor(10)))
                assert(pred_values_exp.device.type==self.device) #################################
                assert(values_exp.device.type==self.device) #################################

                # Compute loss
                # loss = self.loss_fn(pred_values_exp, values_exp)
                loss = t.e**(-self.max_value) * self.loss_fn(pred_values_exp, values_exp) # Added 1/max_value
                # print(f'Predicted value is: {pred_values[0].item()}')
                # print(f'True value is:      {values_batch.view(-1, 1)[0].item()}')
                # print(f'Exp pred value is:  {pred_values_exp[0].item()}')
                # print(f'Exp true value is:  {values_exp[0].item()}')
                # print(f'Loss is:            {loss.item()}')
                # print(f'Max value is:       {self.max_value}')

                assert(loss.device.type==self.device) #################################
                

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Update weights
                self.optimizer.step()

            if (epoch+1) % 10 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

    def prediction_error(self, input_vector, value):
        self.eval()
        with t.no_grad():
            pred_value = self(input_vector)
            exp_pred, exp_value = t.exp(pred_value), t.exp(value)
            # print(exp_pred,exp_value, self.loss_fn(exp_pred,exp_value))
            assert(self.loss_fn(exp_pred, exp_value).device.type==self.device) #################################
            return self.loss_fn(exp_pred, exp_value)

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = "nn-weights" + self.nn_data.file_name[7:-3] + "pt"
        scripted_model = t.jit.script(self)
        scripted_model.save(file_path)



    


#%%

def main(file_name, nn_save_path):
    data = NN_Data(file_name)
    nn = Net(data, epochs=10)
    nn.train_model()
    if nn_save_path is not None:
        nn.save_model(nn_save_path)
    else:
        nn.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a neural network.')
    parser.add_argument('--samples', type=str, required=True, help='Path to the input file')
    parser.add_argument('--nn_path', type=str, default=None, help='Path to save the trained neural network')
    parser.add_argument('--done_path', type=str, default=None, help='Path to save the training done indicator file')
    args = parser.parse_args()
    main(args.samples, args.nn_path)
    # Indicate training is done by creating a status file
    if args.done_path == None:
        done_path = 'training-complete-'+args.samples[8:-4]+'.txt'
    else:
        done_path = args.done_path
    with open(done_path, 'w') as f:
        f.write('Training complete')


