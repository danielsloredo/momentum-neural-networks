from neural_network.ann_base_class import Layer
import numpy as np

class FullyConnectedLayer(Layer): 
    
    def __init__(self, input_size, output_size, bias = True):
        self.weights = np.random.rand(input_size, output_size) - .5
        self.weights_extend = np.zeros((input_size, output_size))
        self.weights_change = np.zeros((input_size, output_size))

        if bias == True:
            self.bias = np.random.rand(1, output_size)  - .5
        else: 
            self.bias = np.zeros((1, output_size))
        
        self.bias_extend = np.zeros((1, output_size))
        self.bias_change = np.zeros((1, output_size))
        
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

    def forward_propagation_momentum(self, input_matrix, gamma_momentum):
        self.input = input_matrix
        self.weights_extend = self.weights + gamma_momentum * self.weights_change
        self.bias_extend = self.bias + gamma_momentum * self.bias_change
        self.output = np.dot(self.input, self.weights_extend) + self.bias_extend
        return self.output

    def backward_propagation_momentum(self, output_derivative, learning_rate, beta_momentum):
        input_derivative = np.dot(output_derivative, self.weights_extend.T)
        weights_derivative = np.dot(self.input.T, output_derivative)

        self.weights_change = beta_momentum * self.weights_change - learning_rate * weights_derivative
        self.weights = self.weights + self.weights_change

        self.bias_change = beta_momentum * self.bias_change - learning_rate * output_derivative
        self.bias = self.bias + self.bias_change
        
        return input_derivative
