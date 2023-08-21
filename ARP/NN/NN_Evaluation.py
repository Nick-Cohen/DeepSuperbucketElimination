#%%
import matplotlib.pyplot as plt
import os
import NN_Train
print(NN_Train.__file__)
import importlib
importlib.reload(NN_Train)
from NN_Train import Net, NN_Data
import torch as t

def evaluate_error_as_function_of_table_value_size(nn: Net, data_test: NN_Data, log_scale = True):
    X = []
    Y = []
    for i in range(len(data_test.input_vectors)):
        x = t.exp(data_test.values[i])
        y = nn.prediction_error(data_test.input_vectors[i],data_test.values[i])
        if log_scale:
            X.append(t.log(x))
            Y.append(t.log(y))
            x_label, y_label = 'log table value', 'log absolute error'
        else:
            X.append(x)
            Y.append(y)
            x_label, y_label = 'table value', 'absolute error'
    plt.scatter(X,Y)
    plt.title('Table Value Versus Absolute Error')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def evaluate_error_by_variable_values(nn: Net, data_test: NN_Data, log_scale = False, show_plot = True):
    nn.eval()  # Put the model in evaluation mode

    # Initialize vectors to accumulate sums
    sum_input_vectors = t.zeros_like(data_test.input_vectors[0])
    sum_input_vectors_times_loss = t.zeros_like(data_test.input_vectors[0])

    # Loop through all the samples in the test data
    for i in range(data_test.num_samples):
        input_vector = data_test.input_vectors[i]
        actual_value = data_test.values[i]

        # Compute the prediction error for this sample
        error = nn.prediction_error(input_vector, actual_value)

        if log_scale:
            # This is not logically correct to do, since summing doesn't make sense
            error = t.log(error)

        # Add the input vector to the total sum of input vectors
        sum_input_vectors += input_vector

        # Add the product of the input vector and the error to the total sum
        sum_input_vectors_times_loss += input_vector * error

    output = sum_input_vectors_times_loss / sum_input_vectors
    if show_plot:
        indices = list(range(len(output)))
        plt.bar(indices, output)
        plt.show()
    return output


def table_histogram(values):
    plt.hist(values, bins=30, edgecolor='black')
    plt.title('Table Values Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()

def test_load_from_jit(savedNN, data: NN_Data):
    model = t.jit.load(savedNN)

    # Print column labels
    print("{:<12} {:<12}".format("Predict", "TrueVal"))

    for i in range(len(data.input_vectors)):
        columns = []
        if data.transform_data:
            prediction_t, true_value_t = float(data.reverse_transform(model(data.input_vectors[i]))), float(data.reverse_transform(data.values[i]))
            columns.append([true_value_t])
            columns.append([prediction_t])
        prediction, true_value = float(model(data.input_vectors[i])), float(data.values[i])
        columns.append([true_value])
        columns.append([prediction])
        
        # Zip the columns, sort by the first column, then unzip
        sorted_columns = list(zip(*sorted(zip(*columns), key=lambda x: x[0])))

        # Convert the tuples back to lists
        sorted_columns = [list(column) for column in sorted_columns]
        
        # Transpose sorted_columns to iterate by rows
        rows = list(zip(*sorted_columns))

        # Print each row with 8 significant digits for each element, using a fixed width of 12 characters
        for row in rows:
            print(' '.join('{:<12.8g}'.format(item) for item in row))



#%%
samples_path = '/home/cohenn1/SDBE/Super_Buckets/BESampling/samples-25;36;47.xml'
data_train = NN_Data(samples_path, device='cpu', transform_data=True)
data_test = NN_Data(samples_path, device='cpu', transform_data=True)
nn = Net(data_train, epochs = 1000)
nn.train_model(batch_size=1000)
nn.save_model('test25;36;47.jit')
# %%
jit_path = '/home/cohenn1/SDBE/Super_Buckets/ARP/NN/test25;36;47.jit'
test_load_from_jit(jit_path, data_train)
#%%


# #%%
# evaluate_error_as_function_of_table_value_size(nn, data_test)
# # %%
# table_histogram(t.exp(data_test.values))
# # %%
# err_by_var = evaluate_error_by_variable_values(nn,data_test, show_plot=True)
# # %%
# # %%
# data_train = NN_Data('/home/cohenn1/SDBE/Super_Buckets/BESampling/samples-202.xml',device='cpu')
# #%%
# # %%

# %%
