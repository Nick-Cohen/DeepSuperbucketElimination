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
import time
from typing import List, Dict, Any, Tuple, IO
import argparse

class NN_Data:

    def __init__(self, file_name = None, processed_samples = None, values = None, device = 'cpu', transform_data = False):
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
        self.transform_data = transform_data
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
        
        # Transform or normalize the data
        if self.transform_data:
            # values -> 10^(values - max value) then normalize to mean 1

            # find max value
            self.max_value = max(self.values)
            # print('max_value is ', max_value)
            
            # subtract max value
            self.values = self.values - self.max_value
            # print('after max subtraction, values_tensor[0] is ', self.values[0])
            
            # exponentiate base 10
            self.values = t.pow(10, self.values)
            # print('after exp subtraction, values_tensor[0] is ', self.values[0])
            
            # normalize to mean 1
            self.mean_transformation_constant = t.mean(self.values)
            print('mean is ', self.mean_transformation_constant)
            self.values = t.div( self.values, self.mean_transformation_constant)
            print('self.values[0:10] is ', self.values[0:10])
            
        self.is_positive = self.values.ne(float('-inf'))
        

        # Get 'outputfnvariabledomainsizes' attribute and convert it to a list of integers
        domain_sizes = [int(x) for x in root.get('outputfnvariabledomainsizes').split(';')]
        self.domain_sizes = t.IntTensor(domain_sizes).to(self.device)
        self.input_vectors = (self.one_hot_encode(self.signatures).float()).to(self.device)
        
    def reverse_transform(self, tensor):
        output = tensor * self.mean_transformation_constant
        output = t.log10(tensor)
        output = output + self.max_value
        output[t.isnan(output)] = 0
        return output



    def one_hot_encode(self, signatures: t.IntTensor, lower_dim = True):
        # transforms (num_samples, num_vars) tensor to (num_samples, sum(domain_sizes)) one hot encoding

        num_samples, num_vars = signatures.shape
        print(num_vars, " input variables.")

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
        hidden_size = 100 # Debug
        # print('Hidden size is ', hidden_size)
        output_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        # self.bn1 = nn.BatchNorm1d(hidden_size).to(self.device)  # Batch Norm layer
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        # self.bn2 = nn.BatchNorm1d(hidden_size).to(self.device)  # Batch Norm layer
        self.fc3 = nn.Linear(hidden_size, output_size).to(self.device)
        
        self.activation_function = nn.Softplus().to(self.device)
        # self.activation_function = nn.ReLU().to(self.device)
 
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.L1Loss()
        self.optimizer = t.optim.Adam(self.parameters())

    def forward(self, x):
        out = self.fc1(x)
        # out = self.bn1(out)
        out = self.activation_function(out)
        out = self.fc2(out)
        # out = self.bn2(out)
        out = self.activation_function(out)
        out = self.fc3(out)
        return out
    
    def train_model(self, batch_size=256):
        if self.num_samples < 256:
            batch_size = self.num_samples // 10
        t = time.time()
        epochs = self.epochs
        X=self.nn_data.input_vectors
        assert(X.device.type==self.device) ############################################################
        num_batches = int(self.num_samples // batch_size)
        # print(self.num_samples, batch_size, num_batches)

        previous_loss = 10e10
        early_stopping_counter = 0
        for epoch in range(epochs):
            for i in range(num_batches):
                # Get mini-batch
                X_batch = X[i*batch_size : (i+1)*batch_size]
                values_batch = self.values[i*batch_size : (i+1)*batch_size]

                # Predict
                pred_values = self(X_batch)


                # Convert predictions and values from log space to original scale minus max value
                # pred_values_exp_adjusted = t.exp(pred_values - self.max_value * t.log(t.tensor(10))) # I might need to change the order I do these exponentiations for numerical precision reasons
                # values_exp_adjusted = t.exp(values_batch.view(-1, 1) - self.max_value * t.log(t.tensor(10)))
                
                # # Convert predictions and values from log space to original scale
                # pred_values_exp = t.exp(pred_values * t.log(t.tensor(10))) # I might need to change the order I do these exponentiations for numerical precision reasons
                # values_exp = t.exp(values_batch.view(-1, 1) * t.log(t.tensor(10)))

                # Compute loss
                loss = self.loss_fn(pred_values, values_batch.view(-1, 1))
                
                # # Custom loss that is weighted by true value
                # mse_loss = nn.functional.mse_loss(pred_values, values_batch.view(-1, 1), reduction='none')
                # custom_loss = mse_loss * values_batch.view(-1, 1)
                # loss = custom_loss.mean()

                
                
                
                # print(pred_values_exp_adjusted)
                # print(loss)
                # stop
                
                # Absolute error
                # loss = self.loss_fn(pred_values_exp, values_exp)
                
                # Absolute error times 1/max value in non log space
                # loss = t.e**(-self.max_value) * self.loss_fn(pred_values_exp, values_exp) # Added 1/max_value
                
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
                # Early stopping
                if (previous_loss - loss.item()) / loss.item() < 0.01:
                # if False:
                    early_stopping_counter += 1
                    if early_stopping_counter >= 10:
                        print("Early stopping triggered.")
                        print("Train time is ", time.time() - t)
                        return # if loss is less than a 1% improvement
                previous_loss = loss
                # Print info
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
        print("Train time is ", time.time() - t)

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

class NetUntransformed(Net):
    
    def __init__(self, net: Net):
        super(NetUntransformed, self).__init__(nn_data=net.nn_data)
        self.net_untransformed = net
        self.nn_data = net.nn_data
        self.mean_transformation_constant = self.nn_data.mean_transformation_constant
        self.max_value = self.nn_data.max_value
    
    # def reverse_transform(self, tensor):
    #     return self.net_untransformed.nn_data.reverse_transform()
    def reverse_transform(self, tensor):
        
        output = tensor * self.mean_transformation_constant
        output = t.log10(tensor)
        output = output + self.max_value
        output[t.isnan(output)] = 0
        return output
        

    def forward(self, x):
        out = self.net_untransformed(x)
        out = self.reverse_transform(out)
        return out
    

    def save_model(self, file_path=None):
        if file_path is None:
            file_path = "nn-weights" + self.nn_data.file_name[7:-3] + "pt"
        scripted_model = t.jit.script(self)
        scripted_model.save(file_path)

    


#%%

def main(file_name, nn_save_path):
    data = NN_Data(file_name, transform_data=False, device='cpu')
    nn = Net(data, epochs=1000)
    nn.train_model()
    if data.transform_data:
        nn = NetUntransformed(nn)
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


