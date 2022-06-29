import numpy as np
import matplotlib.pyplot as plt

from neural_network.ann_neural_network import NeuralNetwork
from neural_network.ann_fully_connected_layer import FullyConnectedLayer
from neural_network.ann_activation_layer import ActivationLayer
from neural_network.ann_activation_functions import leaky_relu, sigmoid, sigmoid_derivative, tanh, tanh_derivative, relu, relu_derivative, leaky_relu_derivative
from neural_network.ann_objective_functions import cross_entropy, cross_entropy_grad, mean_squared_error, mean_squared_error_grad

# training data
x_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0], [1], [1], [0]])

# network
np.random.seed(2)
model_1 = NeuralNetwork()
model_1.add_layer(FullyConnectedLayer(2, 3))
model_1.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))
model_1.add_layer(FullyConnectedLayer(3, 1, bias = False))
model_1.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

# train
model_1.set_objective(mean_squared_error, mean_squared_error_grad)
model_1.fit_momentum(x_train, y_train, epochs=1000, learning_rate=0.001, beta_momentum=.9, gamma_momentum=0)

# test
out = model_1.predict(np.array([[0,0], [0,1], [1,0], [1,1]]))

# network
np.random.seed(2)
model_2 = NeuralNetwork()
model_2.add_layer(FullyConnectedLayer(2, 3))
model_2.add_layer(ActivationLayer(leaky_relu, leaky_relu_derivative))
model_2.add_layer(FullyConnectedLayer(3, 1, bias = False))
model_2.add_layer(ActivationLayer(sigmoid, sigmoid_derivative))

# train
model_2.set_objective(mean_squared_error, mean_squared_error_grad)
model_2.fit_momentum(x_train, y_train, epochs=1000, learning_rate=0.001, beta_momentum=.9, gamma_momentum=.9)

# test
out_2 = model_1.predict(np.array([[1,0], [1,1]]))


print(out)

plt.plot(model_1.objective_values, color = 'blue')
plt.plot(model_2.objective_values, color = 'green')
plt.show()
