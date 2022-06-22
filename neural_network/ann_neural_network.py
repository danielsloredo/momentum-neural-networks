import numpy as np

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

    def create_mini_batches(self, X, y, batch_size):
        mini_batches = []
        n_features = X.shape[1]
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // batch_size
        i = 0
 
        for i in range(n_minibatches):
            mini_batch = data[i * batch_size:(i + 1)*batch_size, :]
            X_mini = mini_batch[:, :n_features]
            Y_mini = mini_batch[:, n_features:]
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % batch_size != 0:
            mini_batch = data[n_minibatches * batch_size:data.shape[0]]
            X_mini = mini_batch[:, :n_features]
            Y_mini = mini_batch[:, n_features:]
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def fit_sgd(self, x_train, y_train, epochs, batch_size, learning_rate):

        for i in range(epochs):
            mini_batches = self.create_mini_batches(x_train, y_train, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch 
                output = X_mini.T

                for layer in self.layers: 
                    output = layer.forward_propagation(output)

                objective_val = self.objective(y_mini.T, output)

                objective_gradient = self.objective_grad(y_mini.T, output)

                for layer in reversed(self.layers):
                    objective_gradient = layer.backward_propagation(objective_gradient, learning_rate)
                
                self.objective_values.append(objective_val)

            print('epoch %d/%d   objective function value = %f' % (i+1, epochs, objective_val))

    def fit_smomentum(self, x_train, y_train, epochs, batch_size, learning_rate, beta_momentum, gamma_momentum):

        for i in range(epochs):
            mini_batches = self.create_mini_batches(x_train, y_train, batch_size)
            for mini_batch in mini_batches:
                X_mini, y_mini = mini_batch 
                output = X_mini.T 

                for layer in self.layers: 
                    output = layer.forward_propagation_momentum(output, gamma_momentum)

                objective_val = self.objective(y_mini.T, output)

                objective_gradient = self.objective_grad(y_mini.T, output)

                for layer in reversed(self.layers):
                    objective_gradient = layer.backward_propagation_momentum(objective_gradient, learning_rate, beta_momentum)

                self.objective_values.append(objective_val)

            print('epoch %d/%d   objective function value = %f' % (i+1, epochs, objective_val))
