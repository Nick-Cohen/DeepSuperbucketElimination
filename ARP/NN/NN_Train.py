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

    def __init__(self, file_name = None, processed_samples = None, values = None, device = 'cpu'):
        self.file_name = file_name
        # self.num_samples: int
        # self.features_per_sample: int
        # self.is_log_space: bool
        # self.max: float
        # self.min: float
        # self.sum: float # log sum of non-logspace values
        # self.signatures: t.Tensor
        # self.input_vectors: t.Tensor
        # self.values: t.Tensor
        # self.is_positive: t.BoolTensor # (num_samples,1) indicates if table has non-zero value: True or zero: False
        # self.domain_sizes: t.IntTensor
        self.device = device
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
        # values_tensor[values_tensor == float('-inf')] = -1e10
        
        self.max_value = float(max(values_tensor)) # take max value exponentiated out of log space

        self.signatures, self.values = signatures_tensor, values_tensor
        # self.values[self.values == float('-inf')] = -1e10
            
        self.is_positive = self.values.ne(float('-inf'))
        

        # Get 'outputfnvariabledomainsizes' attribute and convert it to a list of integers
        domain_sizes = [int(x) for x in root.get('outputfnvariabledomainsizes').split(';')]
        self.domain_sizes = t.IntTensor(domain_sizes).to(self.device)
        # self.input_vectors = (self.one_hot_encode(self.signatures).float()).to(self.device)
        
        # Split into test and validation
        total_samples = self.num_samples
        split_point = min(5000, math.ceil(0.8 * total_samples))  # min of 20% or 5000 for validation

        # Splitting the data into training and test sets
        self.signatures = signatures_tensor[:split_point]
        self.values = values_tensor[:split_point]
        self.input_vectors = self.one_hot_encode(self.signatures).float().to(self.device)[:split_point]
        
        self.signatures_test = signatures_tensor[split_point:]
        self.values_test = values_tensor[split_point:]
        self.input_vectors_test = self.one_hot_encode(self.signatures_test).float().to(self.device)
        
        # Update the number of samples for both training and test sets
        # self.remove_duplicates()
        self.num_samples = len(self.input_vectors)
        self.num_samples_test = len(self.input_vectors_test)

    def remove_duplicates(self):
        def hash_sample(sample):
            return hash(tuple(sample.tolist()))

        # Hashing the training data
        train_hashes = set(hash_sample(sample) for sample in self.signatures)

        # Filtering out duplicates from test data
        unique_indices = [i for i, test_sample in enumerate(self.input_vectors_test)
                          if hash_sample(test_sample) not in train_hashes]

        self.values_test = self.values_test[unique_indices]
        self.input_vectors_test = self.input_vectors_test[unique_indices]
        
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
        self.has_constraints = float('-inf') in nn_data.values
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
        self.parameters_v = list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())
        
        if self.has_constraints:
            self.test_values_filtered = self.nn_data.values_test[self.nn_data.values_test != float('-inf')]
            self.test_X_filtered = self.nn_data.input_vectors_test[self.nn_data.values_test != float('-inf')]
            self.classifier_fc1 = nn.Linear(input_size, hidden_size).to(self.device)
            self.classifier_fc2 = nn.Linear(hidden_size, hidden_size).to(self.device)
            self.classifier_fc3 = nn.Linear(hidden_size, output_size).to(self.device)
            self.parameters_c = list(self.classifier_fc1.parameters()) + list(self.classifier_fc2.parameters()) + list(self.classifier_fc3.parameters())
            self.classifier_values = (~t.isinf(self.values)).float()
            self.classifier_values_test = (~t.isinf(self.nn_data.values_test.float().to(self.device))).float()
            self.es_classifier = False # initialize early stopping condition
            self.es_val_predictor = False
        # Initialize classifier attributes to dummy fc1, this is probably not efficient
        if not self.has_constraints:
            # dummy layers that aren't used but necessary for serialization
            self.classifier_fc1 = nn.Linear(1, 1).to(self.device)
            self.classifier_fc2 = nn.Linear(1, 1).to(self.device)
            self.classifier_fc3 = nn.Linear(1, 1).to(self.device)
            # self.classifier_fc1 = t.tensor(1)
            # self.classifier_fc2 = t.tensor(1)
            # self.classifier_fc3 = t.tensor(1)
        
        self.activation_function = nn.Softplus().to(self.device)
 
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = self.nbe_loss
        # self.loss_fn = self.klish_loss
        if self.has_constraints:
            self.loss_fn2 = nn.BCEWithLogitsLoss() # delete this eventually
            self.loss_fn_c = nn.BCEWithLogitsLoss()
        
        # self.optimizer = t.optim.SGD(self.parameters(), lr=0.01)
        # self.optimizer = t.optim.Adam(self.parameters())
        self.optimizer = t.optim.Adam(self.parameters_v)
        if self.has_constraints:
            self.optimizer_c = t.optim.Adam(self.parameters_c)
        
        # Inialize validation loss twice previous and once previous, used for early stopping
        self.val_loss2 = sys.float_info.max
        self.val_loss1 = self.val_loss2/2
        if self.has_constraints:
            self.val_loss2_c = sys.float_info.max
            self.val_loss1_c = self.val_loss2_c/2

    def forward(self, x, force_positive=t.tensor(False)):
        # value predictor
        out = self.fc1(x)
        out = self.activation_function(out)
        out = self.fc2(out)
        out = self.activation_function(out)
        out = self.fc3(out)
        # binary classifier
        # debug = False
        # if debug:
        if self.has_constraints and not force_positive:
            out2 = self.classifier_fc1(x)
            out2 = self.activation_function(out2)
            out2 = self.classifier_fc2(out2)
            out2 = self.activation_function(out2)
            out2 = self.classifier_fc3(out2)
            out2 = t.sigmoid(out2)
            out2 = t.round(out2) # Masked Net
            # out2 = t.log10(out2)
            neg_inf = t.tensor(float('-inf')).to(out2.device)
            out2 = t.where(out2 == 0, neg_inf, t.tensor(0.))
            # print('out dtype =', out.dtype)
            # print('out2 dtype =', out2.dtype)
            # print(f'out2 is {out2}')
            # sys.exit(1)
            out = out + out2              
        return out
    
    def forward_classifier(self, x):
        out = self.classifier_fc1(x)
        out = self.activation_function(out)
        out = self.classifier_fc2(out)
        out = self.activation_function(out)
        out = self.classifier_fc3(out)
        return out
    
    def forward_train_with_constraints(self, x_filtered, x_unfiltered):
        # value predictor
        out = self.fc1(x_filtered)
        out = self.activation_function(out)
        out = self.fc2(out)
        out = self.activation_function(out)
        out = self.fc3(out)
        # binary classifier
        out2 = self.classifier_fc1(x_unfiltered)
        out2 = self.activation_function(out2)
        out2 = self.classifier_fc2(out2)
        out2 = self.activation_function(out2)
        out2 = self.classifier_fc3(out2)
        return out, out2
    
    def validate_model(self, print_loss=True, type=""):
        self.eval()  # set the model to evaluation mode
        with t.no_grad():
            if not self.has_constraints:
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
                    if math.isnan(avg_val_loss):
                        print("total_val_loss is ", total_val_loss, "num_test_samples is ", num_test_samples)
                return avg_val_loss
            else:
                num_test_samples_classifier = self.nn_data.num_samples_test
                num_test_samples_filtered = len(self.test_values_filtered)
                total_val_loss_classifier = 0
                total_val_loss_filtered = 0
                val_predictions, binary_predictions = self.forward_train_with_constraints(self.test_X_filtered, self.nn_data.input_vectors_test)
                val_loss_classifier = self.loss_fn2(binary_predictions, self.classifier_values_test.view(-1, 1))
                val_loss_filtered = self.loss_fn(val_predictions, self.test_values_filtered.view(-1, 1))
                # nn.BCEWithLogitsLoss(pred_constraint, constraints_values_batch)
                total_val_loss_classifier += val_loss_classifier.item()
                total_val_loss_filtered += val_loss_filtered.item()
                avg_val_loss_classifier = total_val_loss_classifier / num_test_samples_classifier
                if avg_val_loss_classifier == float('-inf') or avg_val_loss_classifier == float('inf'):
                    print('inf loss')
                    # print(binary_predictions)
                    print(self.classifier_values_test.view(-1, 1))
                    sys.exit()
                if num_test_samples_filtered == 0:
                    print("num_test_samples_filtered is zero! ", self.nn_data.file_name)
                    avg_val_loss_filtered = 123456789
                else:
                    avg_val_loss_filtered = total_val_loss_filtered / num_test_samples_filtered
                if print_loss:
                    print('Validation Loss Std: {:.12f}'.format(avg_val_loss_filtered), end=' ')
                    print('Validation Loss Consistency: {:.12f}'.format(avg_val_loss_classifier), end=' ')
                return avg_val_loss_filtered, avg_val_loss_classifier

    def klish_loss(self, y_hat, y):
        # print(t.sum(t.log10(t.mean((y_hat - self.max_value)**10))  + self.max_value))
        # exit(1)
        # return t.sum((y-y_hat)**2)
        return t.sum((y-y_hat)**2) #+ t.sum(t.log10(t.mean((y_hat - self.max_value)**10))  + self.max_value)
    
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
        # use other train method
        if self.has_constraints:
            self.train_model_with_constraints(X,Y,batch_size=batch_size)
            return

        if self.num_samples < 256:
            batch_size = self.num_samples // 10
        tiempo = time.time()
        epochs = self.epochs
        num_batches = int(self.num_samples // batch_size)
        # print(epochs,num_batches)
        previous_loss = 10e10
        early_stopping_counter = 0
        batches = []
        # # Get mini-batch
        for i in range(num_batches):
            X_batch = X[i*batch_size : (i+1)*batch_size]
            values_batch = Y[i*batch_size : (i+1)*batch_size]
            if not self.has_constraints:
                batches.append((X_batch,values_batch))
                
        for epoch in range(epochs):
            for i in range(num_batches):
                X_batch, values_batch = batches[i]

                # Predict value
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
                    print("Train time is ", time.time() - tiempo)
                    self.validate_model(batch_size)
                    self.display_debug_info()
                    return
                else:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
        print("Train time is ", time.time() - tiempo)
        self.display_debug_info()
        self.validate_model(batch_size)
        
    def train_model_with_constraints(self, X, Y, batch_size=2048):
        if self.num_samples < 256:
            batch_size = self.num_samples // 10
        tiempo = time.time()
        epochs = self.epochs
        previous_loss = 10e10
        early_stopping_counter = 0
        # data for predicting the specific value
        X_v = X[Y != float('-inf')]
        Y_v = Y[Y != float('-inf')]
        # deta for the classifier
        X_c = X
        Y_c = self.classifier_values
        # Get batch size
        num_batches_v = int(len(X_v) // batch_size)
        num_batches_c = int(self.num_samples // batch_size)
        print(f'num_batches_v is {num_batches_v} and num_batches_c is {num_batches_c} and len(X_v) is {len(X_v)} and len(X_c) is {len(X_c)}')
        # Get mini-batches
        batches_v = []
        batches_c = []
        for i in range(num_batches_v):
            X_batch = X_v[i*batch_size : (i+1)*batch_size]
            values_batch = Y_v[i*batch_size : (i+1)*batch_size]
            batches_v.append((X_batch,values_batch))
        for i in range(num_batches_c):
            X_batch = X_c[i*batch_size : (i+1)*batch_size]
            values_batch = Y_c[i*batch_size : (i+1)*batch_size]
            batches_c.append((X_batch,values_batch))
            
        early_stop_v = False
        early_stop_c = False
        
        loss_v = t.tensor(float('inf'))
        loss_c = t.tensor(float('inf'))
        
        # Train loop
        for epoch in range(epochs):
            for i in range(num_batches_v):
                if self.es_val_predictor:
                    break
                self.optimizer.zero_grad()
                # self.optimizer_c.zero_grad()
                x_b, y_b = batches_v[i]
                pred = self.forward(x_b, force_positive=True)
                loss_v = self.loss_fn(pred, y_b.view(-1,1))
                
                # Compute gradient via backprop
                loss_v.backward()
                # if epoch % 10 == 0 and i == 0:
                #     print(f'Loss is {loss_v.item()}')
                
                # Update weights
                self.optimizer.step()
                
                # for name, parameter in self.named_parameters():
                #     print(f"{name} gradient: \n{parameter.grad}")
                # sys.exit(1)
                
                
                
                
                
            for i in range(num_batches_c):
                if self.es_classifier:
                    break
                self.optimizer.zero_grad()
                self.optimizer_c.zero_grad()
                x_b, y_b = batches_c[i]
                logits = self.forward_classifier(x_b)
                loss_c = self.loss_fn_c(logits, y_b.view(-1,1))
                
                # Compute gradient via backprop
                loss_c.backward()
                
                # if epoch == 10:
                #     for name, parameter in self.named_parameters():
                #         print(f"Gradient of {name} is {parameter.grad}")
                #     sys.exit(1)
                # print("Before")
                # for name, param in self.named_parameters():
                #     print(f"{name}: {param}")
                # Update weights
                self.optimizer_c.step()

                # print("After")
                # for name, param in self.named_parameters():
                #     print(f"{name}: {param}")
                    
                # sys.exit(1)
                
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch: {epoch+1}')
                # Early stopping
                if self.early_stopping():
                    print("Early stopping triggered.")
                    print("Train time is ", time.time() - tiempo)
                    self.validate_model(batch_size)
                    self.display_debug_info()
                    return
                if self.has_constraints:
                    # print('Epoch [{}/{}], Loss: {:.4f}, XEntropy loss: {:.4f}'.format(epoch+1, epochs, loss_v.item(), loss_c.item()))
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss_v.item()))
                else:
                    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss_v.item()))
        print("Train time is ", time.time() - tiempo)
        self.display_debug_info()
        self.validate_model(batch_size)
        
    def early_stopping(self):
        if not self.has_constraints:
            current_validation_loss = self.validate_model(print_loss=False)
            if self.val_loss2 < self.val_loss1 and self.val_loss1 < current_validation_loss:
                return True
            else:
                self.val_loss2 = self.val_loss1
                self.val_loss1 = current_validation_loss
                return False
        else:
            current_validation_loss, current_validation_loss_c = self.validate_model(print_loss=False)
            # print(f"val_loss2: {self.val_loss2}, val_loss1: {self.val_loss1}, current_validation_loss: {current_validation_loss}, val_loss2_c: {self.val_loss2_c}, val_loss1_c: {self.val_loss1_c}, current_validation_loss_c: {current_validation_loss_c}")
            # if self.val_loss2 < self.val_loss1 and self.val_loss1 < current_validation_loss and self.val_loss2_c < self.val_loss1_c and self.val_loss1_c < current_validation_loss_c:
            
            #early stopping val predictor condition
            if self.val_loss2 < self.val_loss1 and self.val_loss1 < current_validation_loss:
                self.es_val_predictor = True
            #early stopping classifier condition
            if self.val_loss2_c < self.val_loss1_c and self.val_loss1_c < current_validation_loss_c:
                self.es_classifier = True
            
            if self.es_classifier and self.es_val_predictor:
                return True
            else:
                self.val_loss2 = self.val_loss1
                self.val_loss2_c = self.val_loss1_c
                self.val_loss1 = current_validation_loss
                self.val_loss1_c = current_validation_loss_c
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

    def get_full_NN_message(self, full_table_data: NN_Data, model=None):
        input_vectors = t.cat((full_table_data.input_vectors, full_table_data.input_vectors_test), dim=0)
        size = self.nn_data.domain_sizes
        size = tuple(size.tolist())
        if model is None:
            return self(input_vectors).reshape(size)
        else:
            return model(input_vectors).reshape(size)

    def display_debug_info(self):
        print(f'Sample file is {self.nn_data.file_name}')
        print(f'First hundred value predictions are')
        for i in range(100):
            print(f'Predicted: {self.forward(self.nn_data.input_vectors[i], force_positive=False).item()}, True value: {self.nn_data.values[i].item()}')

#%%
def main(file_name, nn_save_path, skip_training=False):
    data = NN_Data(file_name, device='cuda')
    
    gpu = True
    if gpu:
        print('Using GPU')
        nn = Net(data, epochs=1000)
        # nn = Net(data, epochs=1000000, has_constraints=False)
        if not skip_training:
            nn.train_model(data.input_vectors, data.values, batch_size=100)
        else:
            print('SKIPPING TRAINING')
        nn_cpu = Net(data, device='cpu')
        
        # copy weights to cpu model
        model_weights = nn.state_dict()
        cpu_model_weights = {k: v.cpu() for k, v in model_weights.items()}
        nn_cpu.load_state_dict(cpu_model_weights)
    else:
        print('debug test')
        sys.exit()
        nn_cpu = Net(data, epochs=1000000)
        nn_cpu.train_model(data.input_vectors, data.values)
    
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
    main(args.samples, args.nn_path, skip_training=False)
    # Indicate training is done by creating a status file
    if args.done_path == None:
        done_path = 'training-complete-'+args.samples[8:-4]+'.txt'
    else:
        done_path = args.done_path
    with open(done_path, 'w') as f:
        f.write('Training complete')

#%%
# if True:
#     file_name = "/home/cohenn1/SDBE/constraint_problems/samples-251.xml"
#     data = NN_Data(file_name, device='cuda')
#     nn = Net(data, epochs=1000, has_constraints=True)
#     nn.train_model(data.input_vectors, data.values)
# %%
