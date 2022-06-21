class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_matrix):
        raise NotImplementedError

    def backward_propagation(self, output_derivative, learning_rate): 
        raise NotImplementedError 

    def forward_propagation_momentum(self, input_matrix, gamma_momentum):
        raise NotImplementedError

    def backward_propagation_momentum(self, output_derivative, learning_rate, beta_momentum, gamma_momentum):
        raise NotImplementedError

    