from neural_network.ann_base_class import Layer
import numpy as np

class FullyConnectedLayer(Layer): 
    
    def __init__(self, input_size, output_size, bias = True):
        self.weights = np.random.rand(output_size, input_size) - .5
        #self.weights_extend = np.zeros((output_size, input_size))
        self.weights_change = np.zeros((output_size, input_size))

        if bias == True:
            self.bias = np.random.rand(output_size, 1)  - .5
        else: 
            self.bias = np.zeros((output_size, 1))
        
        #self.bias_extend = np.zeros((output_size, 1))
        self.bias_change = np.zeros((output_size, 1))
        
    def forward_propagation(self, input_matrix): 
        self.input = input_matrix
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward_propagation(self, output_derivative, learning_rate):
        weights_derivative = np.dot(output_derivative, self.input.T)
        input_derivative = np.dot(self.weights.T, output_derivative)

        self.weights = self.weights - learning_rate * weights_derivative
        self.bias = self.bias - learning_rate * np.sum(output_derivative, axis=1, keepdims=True)
        return input_derivative

    def forward_propagation_momentum(self, input_matrix, gamma_momentum):
        self.input = input_matrix
        self.weights = self.weights + gamma_momentum * self.weights_change
        self.bias = self.bias + gamma_momentum * self.bias_change
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    def backward_propagation_momentum(self, output_derivative, learning_rate, beta_momentum):
        weights_derivative = np.dot(output_derivative, self.input.T)
        input_derivative = np.dot(self.weights.T, output_derivative)

        self.weights_change = beta_momentum * self.weights_change - learning_rate * weights_derivative
        self.weights = self.weights + self.weights_change

        self.bias_change = beta_momentum * self.bias_change - learning_rate * np.sum(output_derivative, axis = 1, keepdims=True)
        self.bias = self.bias + self.bias_change
        
        return input_derivative
