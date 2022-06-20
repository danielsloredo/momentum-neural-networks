import numpy as np

from neural_network.ann_neural_network import NeuralNetwork
from neural_network.ann_fully_connected_layer import FullyConnectedLayer
from neural_network.ann_activation_layer import ActivationLayer
from neural_network.ann_activation_functions import tanh, tanh_derivative
from neural_network.ann_objective_functions import mean_squared_error, mean_squared_error_grad

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = NeuralNetwork()
net.add_layer(FullyConnectedLayer(2, 3))
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FullyConnectedLayer(3, 1))
net.add_layer(ActivationLayer(tanh, tanh_derivative))

# train
net.set_objective(mean_squared_error, mean_squared_error_grad)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)