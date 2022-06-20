class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_matrix):
        raise NotImplementedError

    def backward_propagation(self, output_derivative, learning_rate): 
        raise NotImplementedError 