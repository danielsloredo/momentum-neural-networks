
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.objective = None
        self.objective_grad = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def set_objective(self, objective, objective_grad):
        self.objective = objective
        self.objective_grad = objective_grad
    
    def predict(self, input_matrix): 
        observations = len(input_matrix)
        result = []

        for i in range(observations): 
            output = input_matrix[i]
            for layer in self.layers: 
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        observations = len(x_train)

        for i in range(epochs): 
            objective_val = 0 

            for j in range(observations): 
                output = x_train[j]

                for layer in self.layers: 
                    output = layer.forward_propagation(output)

                objective_val += self.objective(y_train[j], output)

                objective_gradient = self.objective_grad(y_train[j], output)

                for layer in reversed(self.layers):
                    objective_gradient = layer.backward_propagation(objective_gradient, learning_rate)

            objective_val /= observations

            print('epoch %d/%d   objective function value = %f' % (i+1, epochs, objective_val))

