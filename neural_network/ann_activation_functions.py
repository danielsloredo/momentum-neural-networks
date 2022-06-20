import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    f = sigmoid(x)
    return f * (1 - f)

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def softmax_derivative(x):
    f = softmax(x)
    d_softmax = (f * np.identity(f.size) - f.transpose() @ f)
    return d_softmax

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1-np.tanh(x)**2

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return np.greater(x, 0.).astype(np.float32)

def leaky_relu(x):
    if x>0:
        return x
    else:
        return 0.01*x

def leaky_relu_derivative(x):
    dx = np.ones_like(x)
    dx[x < 0] = 0.01
    return dx