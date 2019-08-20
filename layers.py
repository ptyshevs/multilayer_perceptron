import numpy as np
from activations import Identity, activation_to_obj

def normal_initializer(n, m, loc=0, scale=1):
    return np.random.normal(loc=loc, scale=scale, size=(n, m))

def uniform_initializer(n, m, low=0, high=1):
    return np.random.uniform(low=low, high=high, size=(n, m))

def zero_initializer(n, m):
    return np.zeros((n, m))


class Dense:
    def __init__(self, input_dim, output_dim, activation='identity', trainable=True,
                 dropout_rate=0., initializer='heuristic'):
        """
        Linear -> Activation dense layer
        
        @param activation: activation object (None corresponds to y(x)=x)
        @param trainable: whether to perform update on the backward propagation
        @param dropout_rate: fraction if activations that are zero-ed.
        @param initializer: weights initialization procedure, should be one of those:
                            'he', 'he_unif', 'xavier', 'xavier_unif', 'normal', 'unif', 'heuristic'
                            
                            'heuristic' - if activation is from Relu-family, 'he' is used.
                                          if activation is from sigmoid-family, 'xavier' is used.
                                          otherwise, samples are drawn from ~ N(0, 1) * 0.01
                            bias is always initialized to zero
        """
        self.input_dim, self.output_dim = input_dim, output_dim
        self.activation = self._activation_mapper(activation)
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        self.last_input = None
        
        self._initialize(input_dim, output_dim, initializer)
    
    def forward_propagate(self, X, inference=False):
        Z = X @ self.W.T + self.b.T
        if not inference:
            self.last_input = X
            self.last_output = Z
        
        A = self.activation.forward(Z)
        A = A if (inference or self.dropout_rate == 0.) else self._dropout(A)
        return A
        
    def backward_propagate(self, dA):
#         if self.activation is not None:
#             dZ = self.activation.backward(self.last_output) * dA.T
#         else:
#             dZ = dA.T
        dZ = self.activation.backward(self.last_output) * dA.T
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
    
    def _activation_mapper(self, activation):
        if activation is None:
            return Identity()
        elif type(activation) is str:
            return activation_to_obj(activation)
        else:
            return activation
    
    def _initialize(self, in_dim, out_dim, initializer):
        self.b = zero_initializer(out_dim, 1)

        if initializer == 'heuristic':
            if self.activation == 'relu' or self.activation == 'leaky_relu':
                initializer = 'he'
            elif self.activation == 'sigmoid' or self.activation == 'tanh':
                initializer = 'xavier'
            else:
                initializer = 'normal'

        if initializer == 'he':
            self.W = normal_initializer(out_dim, in_dim, scale=(2/in_dim))
        elif initializer == 'he_unif':
            limit = np.sqrt(6/(in_dim))
            self.W = uniform_initializer(out_dim, in_dim, -limit, limit)
        elif initializer == 'xavier':
            self.W = normal_initializer(out_dim, in_dim, scale=(1/in_dim))
        elif initializer == 'xavier_unif':
            limit = np.sqrt(3/(in_dim))
            self.W = uniform_initializer(out_dim, in_dim, -limit, limit)
        elif initializer == 'normal':
            self.W = normal_initializer(out_dim, in_dim) * .01
        elif initializer == 'unif':
            self.W = uniform_initializer(out_dim, in_dim, -np.sqrt(3),  np.sqrt(3))
        else:
            raise ValueError(f"Unrecognized initialization scheme {initializer}")
            

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