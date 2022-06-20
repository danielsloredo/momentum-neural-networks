from neural_network.ann_base_class import Layer
import numpy as np

class FullyConnectedLayer(Layer): 
    
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - .5
        self.bias = np.random.rand(1, output_size)  - .5
        
    def forward_propagation(self, input_matrix): 
        self.input = input_matrix
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_derivative, learning_rate):
        input_derivative = np.dot(output_derivative, self.weights.T)
        weights_derivative = np.dot(self.input.T, output_derivative)

        self.weights = self.weights - learning_rate * weights_derivative
        self.bias = self.bias - learning_rate * output_derivative
        
        return input_derivative
