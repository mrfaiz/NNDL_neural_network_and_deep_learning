import numpy as np

class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3,1)) - 1
    
    def print_synaptic_weights(self):
        print(self.synaptic_weights)

    def sigmoid(self,x):
        return 1/ (1 + np.exp(-x))
    
    def sigmoid_1st_derivative(self,x):
        return x * (1 - x)

    def train_the_data(self, training_inputs, trianing_expect_output,inerations):

        for i in range(inerations):
            output = self.test_data(training_inputs)

            error = trianing_expect_output - output

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_1st_derivative(output)) 

            self.synaptic_weights += adjustments
    

    def test_data(self,inputs):
        inputs = inputs.astype(float)
        outputs = self.sigmoid(np.dot(inputs,self.synaptic_weights))
        return outputs


training_inputs = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_outputs = np.array([[0,1,1,0]]).T

test_set = np.array([[0,0,1]])

main = NeuralNetwork()

print("Weights before")
main.print_synaptic_weights()

main.train_the_data(training_inputs,training_outputs,100000)

print("Weights After training")
main.print_synaptic_weights()

print("Test set")
print(test_set)

print("output")
print(main.test_data(test_set))