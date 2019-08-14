import numpy as np


def random_initializer(n, m, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n, m) * .01

def zero_initializer(n, m):
    return np.zeros((n, m))


class Dense:
    def __init__(self, input_dim, output_dim, activation=None, trainable=True, dropout_rate=0.):
        """ Linear -> Activation dense layer """
        self.input_dim, self.output_dim = input_dim, output_dim
        self.activation = activation
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        self.last_input = None
        
        self.W = random_initializer(output_dim, input_dim)
        self.b = zero_initializer(output_dim, 1)
    
    def forward_propagate(self, X, inference=False):
        self.last_input = X  # Cache last input
        Z = X @ self.W.T + self.b.T
        self.last_output = Z
        
        A = self.activation.forward(Z) if self.activation is not None else Z
        A = A if (inference or self.dropout_rate == 0.) else self._dropout(A)
        return A
        
    def backward_propagate(self, dA):
        if self.activation is not None:
            dZ = self.activation.backward(self.last_output) * dA.T
        else:
            dZ = dA.T
        dW = self.last_input.T @ dZ / len(dZ)
        db = np.mean(dZ, axis=0, keepdims=True)
        dA_prev = (self.W.T @ dZ.T)
        return dA_prev, dW, db
    
    def _dropout(self, A, correct_magnitude=True):
        mask = np.random.rand(*A.shape) > self.dropout_rate
        A = np.multiply(A, mask)
        if correct_magnitude:
            A = A / (1. - self.dropout_rate)
        return A
    
    def __repr__(self):
        return f"{self.activation} # params = {self._n_params()}"
    
    def _n_params(self):
        w = self.W.shape[0] * self.W.shape[1]
        b = self.b.shape[0]
        return w + b

class Dropout:
    def __init__(self, drop_rate=.5, correct_magnitude=True):
        self.keep_prob = 1 - drop_rate
        self.correct_magnitude = correct_magnitude
    
    def forward_propagate(self, X, inference=False):
        if inference:
            return X
        mask = np.random.rand(*X.shape) < self.keep_prob
        Z = np.multiply(X, mask)
        if self.correct_magnitude:
            Z /= self.keep_prob
        return Z
    
    def backward_propagate(self, dA):
        return dA