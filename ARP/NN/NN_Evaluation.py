#%%
import matplotlib.pyplot as plt
from NN_Train import Net, NN_Data

def evaluate_error_as_function_of_table_value_size(nn, data_test):
    X = []
    Y = []
    for i in range(len(data_test.input_vectors)):
        x = t.exp(data_test.values[i])
        y = nn.prediction_error(data_test.input_vectors[i], x)
        X.append(x)
        Y.append(y)
    plt.scatter(X,Y)
#%%
data_train = NN_Data('../../BESampling/samples-39;104;141.xml')
data_test = NN_Data('../../BESampling/samples-39;104;141-full.xml')
nn = Net(data_train)
nn.train_model()
# %%
evaluate_error_as_function_of_table_value_size(nn, data_test)
# %%
