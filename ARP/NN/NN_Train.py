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
from torch.utils.data import DataLoader, TensorDataset
import xml.etree.ElementTree as ET
import time
from typing import List, Dict, Any, Tuple, IO
import argparse
import math
import sys

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
        self.max_value = None
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
        
        self.max_value = float(max(values_tensor)) # take max value exponentiated out of log space
        # self.max_value = float(t.exp(t.log(t.tensor(10)) * max(values_tensor))) # take max value exponentiated out of log space

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
        # self.input_vectors = (self.one_hot_encode(self.signatures).float()).to(self.device)
        
        # Split into test and validation
        total_samples = self.num_samples
        split_point = math.ceil(0.8 * total_samples)  # 80% for training

        # Splitting the data into training and test sets
        self.signatures = signatures_tensor[:split_point]
        self.values = values_tensor[:split_point]
        self.input_vectors = self.one_hot_encode(self.signatures).float().to(self.device)[:split_point]
        
        self.signatures_test = signatures_tensor[split_point:]
        self.values_test = values_tensor[split_point:]
        self.input_vectors_test = self.one_hot_encode(self.signatures_test).float().to(self.device)
        
        # Update the number of samples for both training and test sets
        self.num_samples = split_point
        self.num_samples_test = total_samples - split_point
        
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

    def __init__(self, nn_data: NN_Data, epochs = 1000, device = None):
        super(Net, self).__init__()
        self.nn_data = nn_data
        self.max_value = nn_data.max_value
        self.num_samples = self.nn_data.num_samples
        self.epochs = epochs
        if device is None:
            self.device = nn_data.device
        else:
            self.device = device
        self.values = nn_data.values.float().to(self.device)
        input_size, hidden_size = len(nn_data.input_vectors[0]), len(nn_data.input_vectors[0])*2
        hidden_size = 100
        output_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(self.device)
        
        self.activation_function = nn.Softplus().to(self.device)
 
        # self.loss_fn = nn.MSELoss()
        # self.loss_fn = self.nbe_loss
        self.loss_fn = self.klish_loss
        
        # self.optimizer = t.optim.SGD(self.parameters(), lr=0.01)
        self.optimizer = t.optim.Adam(self.parameters())
        
        # Inialize validation loss twice previous and once previous, used for early stopping
        self.val_loss2 = sys.float_info.max
        self.val_loss1 = self.val_loss2/2

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation_function(out)
        out = self.fc2(out)
        out = self.activation_function(out)
        out = self.fc3(out)
        return out
    
    def validate_model(self, print_loss=True):
        self.eval()  # set the model to evaluation mode
        with t.no_grad():
            test_X = self.nn_data.input_vectors_test
            test_Y = self.nn_data.values_test.float().to(self.device)
            num_test_samples = self.nn_data.num_samples_test

            total_val_loss = 0
            pred = self(test_X)
            val_loss = self.loss_fn(pred, test_Y.view(-1, 1))
            total_val_loss += val_loss.item()
            avg_val_loss = total_val_loss / num_test_samples
            if print_loss:
                print('Validation Loss: {:.12f}'.format(avg_val_loss))
            return avg_val_loss

    def klish_loss(self, y_hat, y):
        # print(t.sum(t.log10(t.mean((y_hat - self.max_value)**10))  + self.max_value))
        # exit(1)
        # return t.sum((y-y_hat)**2)
        return t.sum((y-y_hat)**2) + t.sum(t.log10(t.mean((y_hat - self.max_value)**10))  + self.max_value)
    
    def nbe_loss(self, y_hat, y):
        # print(t.pow(1.5, y_hat - self.nn_data.max_value))
        # return t.sum(t.pow((y - y_hat),2))
        # print(t.sum(t.abs(y_hat/self.nn_data.max_value) * t.pow((y - y_hat),2)))
        # print(t.pow(1.1, t.max(y,y_hat) - self.nn_data.max_value))
        # exit(1)
        return t.sum(t.pow(1.1, t.max(y,y_hat) - self.nn_data.max_value) * t.pow((y - y_hat),2))
        return t.sum(t.pow(10, y_hat - self.nn_data.max_value) * t.pow((y - y_hat),2))
    
    # adding term to try to punish overestimation bias
    def custom_loss2(self, y_hat, y):
        base = 2.0
        return t.sum((y_hat - y)**2 * base ** (y_hat - y))
        # return t.sum((y_hat - y)**2 * base ** (t.max(y_hat,y) - y))
    
    def train_model(self, X, Y, batch_size=2048):
        if self.num_samples < 256:
            batch_size = self.num_samples // 10
        t = time.time()
        epochs = self.epochs
        num_batches = int(self.num_samples // batch_size)
        print(epochs,num_batches)
        previous_loss = 10e10
        early_stopping_counter = 0
        batches = []
        # # Get mini-batch
        for i in range(num_batches):
            X_batch = X[i*batch_size : (i+1)*batch_size]
            values_batch = Y[i*batch_size : (i+1)*batch_size]
            batches.append((X_batch,values_batch))
        for epoch in range(epochs):
            for i in range(num_batches):
                X_batch, values_batch = batches[i]

                # Predict
                pred_values = self(X_batch)

                # Compute loss
                loss = self.loss_fn(pred_values, values_batch.view(-1, 1))
                
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()

                # Update weights
                self.optimizer.step()

            
            if (epoch+1) % 10 == 0:
                # Early stopping
                if self.early_stopping():
                    print("Early stopping triggered.")
                    print("Train time is ", time.time() - t)
                    self.validate_model(batch_size)
                    return

                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
        print("Train time is ", time.time() - t)
        self.validate_model(batch_size)
        
    def early_stopping(self):
        current_validation_loss = self.validate_model(print_loss=False)
        if self.val_loss2 < self.val_loss1 and self.val_loss1 < current_validation_loss:
            return True
        else:
            self.val_loss2 = self.val_loss1
            self.val_loss1 = current_validation_loss
            return False
            

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
        # self.cpu()
        scripted_model = t.jit.script(self)
        scripted_model.save(file_path)

    
class DummyNet(nn.Module):
    
    def __init__(self, input_vectors, values, device):
        super(DummyNet, self).__init__()
        self.input_vectors = input_vectors
        self.values = values
        self.device = device
        input_size = len(input_vectors[0])
        hidden_size = 100
        output_size = 1
        self.fc1 = nn.Linear(input_size, hidden_size).to(self.device)
        self.fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.fc3 = nn.Linear(hidden_size, output_size).to(self.device)
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x):
        x = self.fc1(x)
        x = t.relu(x)
        x = self.fc2(x)
        x = t.relu(x)
        x = self.fc3(x)
        return x
    
    # def train_model(self, X,Y, epochs=100, lr=0.001):
    #     input_vectors = X
    #     values = Y
            
    #     criterion = nn.MSELoss()
    #     optimizer = t.optim.Adam(self.parameters(), lr=lr)
            
    #     for epoch in range(epochs):
    #         self.train()
    #         optimizer.zero_grad()
                
    #         outputs = self(input_vectors)
    #         loss = criterion(outputs, values)
                
    #         loss.backward()
    #         optimizer.step()
    #         print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    
    def train_model(self, input_vectors, values, epochs=100, lr=0.001, batch_size=256):
        # Convert to PyTorch tensors and move to the specified device
        input_vectors = t.tensor(input_vectors, dtype=t.float32).to(self.device)
        values = t.tensor(values, dtype=t.float32).to(self.device)
        
        # Create a DataLoader
        dataset = TensorDataset(input_vectors, values)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize the optimizer
        optimizer = t.optim.Adam(self.parameters(), lr=lr)

        my_batches = list(dataloader)
        print(my_batches)
        for epoch in range(epochs):
            self.train()
            
            #a,b = next(iter(dataloader))
            #print(a.device)
            for batch_input_vectors, batch_values in my_batches: #[(a,b)]: #dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(batch_input_vectors)
                
                # Compute the loss
                loss = self.loss_fn(outputs, batch_values)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
            
            print(f'Epoch: {epoch+1}/{epochs}, Last batch loss: {loss.item()}')




#%%

def main(file_name, nn_save_path):
    data = NN_Data(file_name, transform_data=False, device='cuda')
    
    gpu = True
    if gpu:
        print('Using GPU')
        nn = Net(data, epochs=1000000)
        nn.train_model(data.input_vectors, data.values)
        nn_cpu = Net(data, device='cpu')
        
        # copy weights to cpu model
        model_weights = nn.state_dict()
        cpu_model_weights = {k: v.cpu() for k, v in model_weights.items()}
        nn_cpu.load_state_dict(cpu_model_weights)
    else:
        nn_cpu = Net(data, epochs=1000000)
        nn_cpu.train_model(data.input_vectors, data.values)
    
    if data.transform_data:
        nn = NetUntransformed(nn)
    if nn_save_path is not None:
        nn_cpu.save_model(nn_save_path)
    else:
        nn_cpu.save_model()

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


