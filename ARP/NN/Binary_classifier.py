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
import NN_Train

def binarize_values(values):
    output = t.ones_like(values)
    output[values]

class Binary_classifier(nn.Module):

    def __init__(self, nn_data, epochs = 1000, device = None):
        super(Binary_classifier, self).__init__()
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
    
    def evaluate_model(self, print_loss=True, save_plot=False, plot_path=None):
        import numpy as np
        self.eval()  # set the model to evaluation mode
        with t.no_grad():
            test_X = self.nn_data.input_vectors_test
            test_Y = self.nn_data.values_test.float().to(self.device)
            
            # Prediction
            pred = self(test_X).view(-1)

            # Calculate and print validation loss
            val_loss = self.loss_fn(pred, test_Y)
            avg_val_loss = val_loss.item() / len(test_Y)
            if print_loss:
                print('Validation Loss: {:.12f}'.format(avg_val_loss))

            # Correlation coefficient
            corr_coeff = np.corrcoef(pred.cpu().numpy(), test_Y.cpu().numpy())[0, 1]
            print('Correlation Coefficient:', corr_coeff)

            # Scatter plot
            if save_plot:
                path = plot_path
                if plot_path is None:
                    path = str(self.nn_data.signatures.tolist()) + ".png"
                import matplotlib.pyplot as plt
                import torch as t
                plt.scatter(pred.cpu().numpy(), test_Y.cpu().numpy())
                plt.xlabel('Predicted Values')
                plt.ylabel('True Values')
                plt.title('Scatter Plot of Predictions vs True Values')
                plt.savefig(plot_path)

        return avg_val_loss, corr_coeff

    
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

    def get_full_NN_message(self, full_table_data: NN_Data, model=None):
        input_vectors = t.cat((full_table_data.input_vectors, full_table_data.input_vectors_test), dim=0)
        size = self.nn_data.domain_sizes
        size = tuple(size.tolist())
        if model is None:
            return self(input_vectors).reshape(size)
        else:
            return model(input_vectors).reshape(size)