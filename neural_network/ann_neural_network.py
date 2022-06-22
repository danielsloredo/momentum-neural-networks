class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.objective = None
        self.objective_grad = None
        self.objective_values = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_objective(self, objective, objective_grad):
        self.objective = objective
        self.objective_grad = objective_grad
    
    def predict(self, input_test): 
        result = input_test.T
        for layer in self.layers: 
            result = layer.forward_propagation(result)
        return result.T

    def fit_gd(self, x_train, y_train, epochs, learning_rate):

        for i in range(epochs): 
            output = x_train.T

            for layer in self.layers: 
                output = layer.forward_propagation(output)

            objective_val = self.objective(y_train.T, output)

            objective_gradient = self.objective_grad(y_train.T, output)

            for layer in reversed(self.layers):
                objective_gradient = layer.backward_propagation(objective_gradient, learning_rate)
                
            self.objective_values.append(objective_val)

            print('epoch %d/%d   objective function value = %f' % (i+1, epochs, objective_val))
    

    def fit_momentum(self, x_train, y_train, epochs, learning_rate, beta_momentum, gamma_momentum):

        for i in range(epochs): 
            output = x_train.T

            for layer in self.layers: 
                output = layer.forward_propagation_momentum(output, gamma_momentum)

            objective_val = self.objective(y_train.T, output)

            objective_gradient = self.objective_grad(y_train.T, output)

            for layer in reversed(self.layers):
                objective_gradient = layer.backward_propagation_momentum(objective_gradient, learning_rate, beta_momentum)

            self.objective_values.append(objective_val)

            print('epoch %d/%d   objective function value = %f' % (i+1, epochs, objective_val))
