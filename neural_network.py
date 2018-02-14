# Importing Libraries
import numpy as np
import glob
import matplotlib as mpimg
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
#%matplotlib inline

# Defining sigmoid and it's derivative functions
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
    
def sigmoid_derivative(x):
    return (x * (1 - x))

# Gettting Paths

path_names = glob.glob('faces/*/*')
len(path_names)

pre_data = []
for path in path_names:   
    img = cv2.imread(path,0)
    #print img.shape
    resized = cv2.resize(img, (30,32))
    pre_data.append(np.concatenate((resized), axis = 0))

data = np.reshape(pre_data, (1888, 30*32))
data.shape

# Each data in 'data' variable contains image and we can construct the images easily. (Shown below for the 1st entry)
print "Example Image:" 
plt.imshow(data[0,:].reshape(32, 30), cmap = 'gray')
plt.show()

labels = list()
for path in path_names:
    if('neutral' in path):
        labels.append(np.array([1,0,0,0]))
    elif('happy' in path):
        labels.append(np.array([0,1,0,0]))
    elif('sad' in path):
        labels.append(np.array([0,0,1,0]))
    elif('angry' in path):
        labels.append(np.array([0,0,0,1]))


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

print "Neural Network, Part 1: Using all training data (Including sunglasses)"

print "Size of training data: ", x_train.shape[0],',', x_train.shape[1]
print "Size of training labels: ", len(y_train)
print "Size of testing data: ", x_test.shape[0], ',', x_test.shape[1]
print "Size of testing labels: ", len(y_test)

print "Traning on Neural nework with 1 hidden layer and 10 nodes:"
# Algorithm
n_hidden = 30
output_nodes = 4
epochs = 100
learn_rate = 0.01

n_records, n_features = x_train.shape

#Randomly Initialize Weights
#weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
#                                        size=(n_features, n_hidden))
#weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
#                                         size=(n_hidden, output_nodes))

weights_input_hidden = np.random.normal(scale=0.1,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=0.1,
                                         size=(n_hidden, output_nodes))


last_loss = None

# Training Phase

for e in tqdm(range(epochs)):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)

    for n in range(len(x_train)):
        x_non_norm = x_train[n]
        x = x_non_norm / 255.0
        y = y_train[n]
        
        ## Forward pass##
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output, weights_hidden_output))

        ## Output error ##
        error = y - output
        output_delta = error * sigmoid_derivative(output)
        #print output_delta
        
        ## Backpropogated error ##
        hidden_error = np.dot(output_delta, weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
        #print hidden_delta
        
        # Delta weights
        del_w_hidden_output += output_delta*hidden_output[:,None]
        del_w_input_hidden += hidden_delta*x[:,None]
        
        weights_input_hidden += learn_rate*del_w_input_hidden
        weights_hidden_output += learn_rate*del_w_hidden_output

print "Testing accuracy on NN with one node: "
#Testing Phase
correct = 0;
for n in range(len(x_test)):
    x_non_norm = x_test[n]
    x = x_non_norm/255.00
    y = y_test[n]

    ## Forward pass##
    hidden_input = np.dot(x, weights_input_hidden)
    #print hidden_input
    hidden_output = sigmoid(hidden_input)
    #print hidden_output
    output_ = sigmoid(np.dot(hidden_output, weights_hidden_output))
    for i in range(4):
        if(output_[i] == np.max(output_)):
            output = np.array([0, 0, 0, 0])
            output[i] = 1
            if(np.mean(output == y) == 1.0):
                correct = correct + 1

print "Testing Accuracy:", correct/len(y), "%"

print "Training Neural Network on 2 hidden layers with 10 nodes each: "
# Training Phase for 2 hidden layers
n_hidden_1 = 10
n_hidden_2 = 10
output_nodes = 4
epochs = 300
learn_rate = 0.01

n_records, n_features = x_train.shape

#Randomly Initialize Weights
weights_input_hidden_1 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden_1))
weights_hidden_1_hidden_2 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_hidden_1, n_hidden_2))
weights_hidden_2_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=(n_hidden_2, output_nodes))

last_loss = None

for e in tqdm(range(epochs)):
    del_w_input_hidden_1 = np.zeros(weights_input_hidden_1.shape)
    del_w_hidden_1_hidden_2 = np.zeros(weights_hidden_1_hidden_2.shape)
    del_w_hidden_2_output = np.zeros(weights_hidden_2_output.shape)

    for n in range(len(x_train)):
        x_non_norm = x_train[n]
        x = x_non_norm / 255.0
        y = y_train[n]
        
        ## Forward pass##
        hidden_input_1 = np.dot(x, weights_input_hidden_1)
        hidden_output_1 = sigmoid(hidden_input_1)
        hidden_input_2 = np.dot(hidden_output_1, weights_hidden_1_hidden_2)
        hidden_output_2 = sigmoid(hidden_input_2)
        output = sigmoid(np.dot(hidden_output_2, weights_hidden_2_output))
        for i in range(4):
            if(output[i] == np.max(output)):
                output_ = np.array([0, 0, 0, 0])
                output_[i] = 1
        #print output_

        ## Output error ##
        error = y - output
        output_delta = error * sigmoid_derivative(output)
        #print output_delta
        
        ## Backpropogated error ##
        hidden_error_2 = np.dot(output_delta, weights_hidden_2_output.T)
        hidden_delta_2 = hidden_error_2 * sigmoid_derivative(hidden_output_2)
        
        hidden_error_1 = np.dot(hidden_delta_2, weights_hidden_1_hidden_2.T)
        hidden_delta_1 = hidden_error_1 * sigmoid_derivative(hidden_output_1)
        
        # Delta weights
        del_w_hidden_2_output = output_delta*hidden_output_2[:,None]
        del_w_hidden_1_hidden_2 = hidden_delta_2*hidden_output_1[:,None]
        del_w_input_hidden_1 = hidden_delta_1*x[:,None]
        
        weights_input_hidden_1 += learn_rate*del_w_input_hidden_1
        weights_hidden_1_hidden_2 += learn_rate*del_w_hidden_1_hidden_2
        weights_hidden_2_output += learn_rate*del_w_hidden_2_output

print "Testing accuracy on NN with 2 nodes: "
# Test Data Set
correct = 0
for n in range(len(x_test)):
    x_non_norm = x_test[n]
    x = x_non_norm / 255.0
    y = y_test[n]

    ## Forward pass##
    hidden_input_1 = np.dot(x, weights_input_hidden_1)
    hidden_output_1 = sigmoid(hidden_input_1)
    hidden_input_2 = np.dot(hidden_output_1, weights_hidden_1_hidden_2)
    hidden_output_2 = sigmoid(hidden_input_2)
    output_ = sigmoid(np.dot(hidden_output_2, weights_hidden_2_output))
    for i in range(4):
        if(output_[i] == np.max(output_)):
            output = np.array([0, 0, 0, 0])
            output[i] = 1
            if(np.mean(output == y) == 1.0):
                correct = correct + 1

print "Testing Accuracy:", correct/len(y), "%"

print "Neural Network, Part 2: Using training data with no sunglasses"

labels_no_glasses = list()
pre_data = []
for path in path_names:
    if ('sunglasses' not in path):  
        img = cv2.imread(path,0)
        resized = cv2.resize(img, (30,32))
        pre_data.append(np.concatenate((resized), axis = 0))
        if('neutral' in path):
            labels_no_glasses.append(np.array([1,0,0,0]))
        elif('happy' in path):
            labels_no_glasses.append(np.array([0,1,0,0]))
        elif('sad' in path):
            labels_no_glasses.append(np.array([0,0,1,0]))
        elif('angry' in path):
            labels_no_glasses.append(np.array([0,0,0,1]))

data = np.reshape(pre_data, (len(pre_data), 30*32))
x_train, x_test, y_train, y_test = train_test_split(data, labels_no_glasses, test_size = 0.3, random_state=42)

print "Size of training data: ", x_train.shape[0],',', x_train.shape[1]
print "Size of training labels: ", len(y_train)
print "Size of testing data: ", x_test.shape[0], ',', x_test.shape[1]
print "Size of testing labels: ", len(y_test)

# Algorithm
n_hidden_1 = 10
n_hidden_2 = 10
output_nodes = 4
epochs = 300
learn_rate = 0.01

n_records, n_features = x_train.shape

#Randomly Initialize Weights
weights_input_hidden_1 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden_1))
weights_hidden_1_hidden_2 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_hidden_1, n_hidden_2))
weights_hidden_2_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=(n_hidden_2, output_nodes))

last_loss = None

print "Training on Neural Network with 2 hidden layers with 10 nodes each"

for e in tqdm(range(epochs)):
    del_w_input_hidden_1 = np.zeros(weights_input_hidden_1.shape)
    del_w_hidden_1_hidden_2 = np.zeros(weights_hidden_1_hidden_2.shape)
    del_w_hidden_2_output = np.zeros(weights_hidden_2_output.shape)

    for n in range(len(x_train)):
        x_non_norm = x_train[n]
        x = x_non_norm / 255.0
        y = y_train[n]
        
        ## Forward pass##
        hidden_input_1 = np.dot(x, weights_input_hidden_1)
        hidden_output_1 = sigmoid(hidden_input_1)
        hidden_input_2 = np.dot(hidden_output_1, weights_hidden_1_hidden_2)
        hidden_output_2 = sigmoid(hidden_input_2)
        output = sigmoid(np.dot(hidden_output_2, weights_hidden_2_output))
        for i in range(4):
            if(output[i] == np.max(output)):
                output_ = np.array([0, 0, 0, 0])
                output_[i] = 1
        #print output_

        ## Output error ##
        error = y - output
        output_delta = error * sigmoid_derivative(output)
        #print output_delta
        
        ## Backpropogated error ##
        hidden_error_2 = np.dot(output_delta, weights_hidden_2_output.T)
        hidden_delta_2 = hidden_error_2 * sigmoid_derivative(hidden_output_2)
        
        hidden_error_1 = np.dot(hidden_delta_2, weights_hidden_1_hidden_2.T)
        hidden_delta_1 = hidden_error_1 * sigmoid_derivative(hidden_output_1)
        
        # Delta weights
        del_w_hidden_2_output = output_delta*hidden_output_2[:,None]
        del_w_hidden_1_hidden_2 = hidden_delta_2*hidden_output_1[:,None]
        del_w_input_hidden_1 = hidden_delta_1*x[:,None]
        
        weights_input_hidden_1 += learn_rate*del_w_input_hidden_1
        weights_hidden_1_hidden_2 += learn_rate*del_w_hidden_1_hidden_2
        weights_hidden_2_output += learn_rate*del_w_hidden_2_output

print "Testing Accuracy on NN with no sunglasses data"

# Test Data Set
correct = 0
for n in range(len(x_test)):
    x_non_norm = x_test[n]
    x = x_non_norm / 255.0
    y = y_test[n]

    ## Forward pass##
    hidden_input_1 = np.dot(x, weights_input_hidden_1)
    hidden_output_1 = sigmoid(hidden_input_1)
    hidden_input_2 = np.dot(hidden_output_1, weights_hidden_1_hidden_2)
    hidden_output_2 = sigmoid(hidden_input_2)
    output_ = sigmoid(np.dot(hidden_output_2, weights_hidden_2_output))
    for i in range(4):
        if(output_[i] == np.max(output_)):
            output = np.array([0, 0, 0, 0])
            output[i] = 1
            if(np.mean(output == y) == 1.0):
                correct = correct + 1

print "Testing Accuracy:", correct/len(y), "%"


