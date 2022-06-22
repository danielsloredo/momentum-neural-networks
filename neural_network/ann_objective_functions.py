import numpy as np

def mean_squared_error(y_obs, y_pred):
    return np.mean(np.power(y_obs-y_pred, 2))

def mean_squared_error_grad(y_obs, y_pred):
    return 2 * (y_pred - y_obs)/y_pred.shape[0]

def cross_entropy(y_obs, y_pred):
    loss=-np.sum(y_obs*np.log(y_pred))
    return loss/float(y_pred.shape[0])

def cross_entropy_grad(y_obs, y_pred):
    m = y_obs.shape[0]
    grad = y_pred - y_obs
    grad = grad/m
    return grad
