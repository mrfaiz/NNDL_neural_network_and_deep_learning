# https://www.youtube.com/watch?v=kft1AJ9WVDk&feature=share&fbclid=IwAR0-GE654jG-MDNBvHlBjleokKH_rWOxUbsz_7502QV7k6voeflLBYExZ7Q

# Inputs     output
# 0 0 1        0
# 1 1 1        1
# 1 0 1        1
# 0 1 1        0 

# new input 

# 1 0 0        ?

# * hints : first input is 1 , output is 1

import numpy as np 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_1st_derivative(x):
    return x * (1 - x)

training_data = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_output = np.array([[0,1,1,0]]).T

# print(training_output)
# print(sigmoid(-3.4))

np.random.seed(1)

# 3,1 matrix fill up with data -1 to +1
synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Synaptic Weights")
print(synaptic_weights)

for iteration in range(100000):
    input_layer = training_data
    outputs = sigmoid(np.dot(input_layer,synaptic_weights))

    error = training_output - outputs

    adjustment = error * sigmoid_1st_derivative(outputs)

    synaptic_weights += np.dot(input_layer.T,adjustment)

print("Weights after training **")
print(synaptic_weights)

print("Output")
print(outputs)
