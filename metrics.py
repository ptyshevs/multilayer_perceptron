import numpy as np

def accuracy(y_true, y_pred):
    ev = y_true == y_pred
    return ev.sum() / ev.size

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))