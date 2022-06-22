import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1 - f)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.greater(x, 0.).astype(np.float32)

def leaky_relu(x):
    return np.where(x > 0, x, x * 0.01)  

def leaky_relu_derivative(x):
    return np.where(x > 0, 1, 0.01)