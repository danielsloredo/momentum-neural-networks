from neural_network.ann_base_class import Layer

class ActivationLayer(Layer):

    def __init__(self, activation_function, activation_derivative): 
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward_propagation(self, input_matrix):
        self.input = input_matrix
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_derivative, learning_rate):
        return output_derivative * self.activation_derivative(self.input)

    def forward_propagation_momentum(self, input_matrix, gamma_momentum):
        self.input = input_matrix
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation_momentum(self, output_derivative, learning_rate, beta_momentum):
        return output_derivative * self.activation_derivative(self.input)