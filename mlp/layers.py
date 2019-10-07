import numpy as np
from .activations import activation_to_obj

def normal_initializer(n, m, d=1, loc=0, scale=1):
    return np.random.normal(loc=loc, scale=scale, size=(n, m, d)).squeeze()

def uniform_initializer(n, m, d=1, low=0, high=1):
    return np.random.uniform(low=low, high=high, size=(n, m, d)).squeeze()

def zero_initializer(n, m, d=1):
    return np.zeros((n, m, d)).squeeze()


class L1Regularizer:
    def __init__(self, lamd=.01):
        self.lamd = lamd
    
    def __call__(self, M):
        # Gradient only
        return self.lamd * np.where(M > 0, 1.0, -1.0)

class L2Regularizer:
    def __init__(self, lamd=.01):
        self.lamd = lamd
    
    def __call__(self, M):
        # This is used only in back-propagation, so here is the gradient
        return self.lamd * M

regularizers_map = {'l1': L1Regularizer(), 'l2': L2Regularizer()}
    
def regularizer_mapper(regularizer):
    if type(regularizer) is str:
        if regularizer in regularizers_map:
            return regularizers_map[regularizer]
        else:
            raise ValueError("Regularizer is unrecognized:", regularizer)
    return regularizer

class Layer:
    def __init__(self, trainable=True):
        self.input_dim, self.output_dim = None, None
        
        self.trainable = trainable
        self.params = []

    def forward_propagate(self):
        pass
    
    def backward_propagate(self):
        pass
    
    def _initialize(self, *args, **kwargs):
        pass
    
    @property
    def _n_params(self):
        return sum([p.size for p in self.params])
    
    def __repr__(self):
        return self.__class__.__name__


class Dense(Layer):
    def __init__(self, n_units, activation='identity', trainable=True,
                 dropout_rate=0., initializer='heuristic', weights_regularizer=None, bias_regularizer=None):
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
        self.input_dim, self.output_dim = None, n_units
        self.activation = self._activation_mapper(activation)
        self.trainable = trainable
        self.dropout_rate = dropout_rate
        self.last_input = None
        
        self.initializer = initializer
        self.weights_regularizer = weights_regularizer
        self.bias_regularizer = bias_regularizer
        self.params = list()

    def forward_propagate(self, X, inference=False):
        Z = X @ self.W.T + self.b.T
        if not inference:
            self.last_input = X
            self.last_output = Z
        
        A = self.activation.forward(Z)
        if not inference and self.dropout_rate > 0:
            A = self._dropout(A)
        return A
        
    def backward_propagate(self, dA):
        # Todo: fix dropout back-propagation: dA should have the same mask applied and scaled by the same keep_prob
        dZ = self.activation.backward(self.last_output) * dA.T
        m = len(dZ)
        dW = self.last_input.T @ dZ / m
        if self.weights_regularizer:
            dW += self.weights_regularizer(self.W.T) / m
        db = np.mean(dZ, axis=0, keepdims=True)
        if self.bias_regularizer:
            db += self.bias_regularizer(self.b) / m
        dA_prev = self.W.T @ dZ.T
        return dA_prev, dW.T, db.T
    
    def _dropout(self, A, correct_magnitude=True):
        mask = np.random.rand(*A.shape) > self.dropout_rate
        A = np.multiply(A, mask)
        if correct_magnitude:
            A = A / (1. - self.dropout_rate)
        return A
    
    def __repr__(self):
        return f"|Dense({self.input_dim}, {self.output_dim}, {self.activation})".ljust(20) + f"\t|\t{self._n_params}".rjust(20)
    
    @property
    def _n_params(self):
        return sum([p.size for p in self.params])
    
    def _activation_mapper(self, activation):
        if activation is None:
            return activation_to_obj('identity')
        elif type(activation) is str:
            return activation_to_obj(activation)
        else:
            return activation
    
    def _initialize(self, in_dim):
        self.params.clear()
        out_dim = self.output_dim
        self.input_dim = in_dim
        initializer = self.initializer

        self.b = zero_initializer(out_dim, 1)

        if initializer == 'heuristic':
            if self.activation == 'relu' or self.activation == 'leaky_relu':
                initializer = 'he'
            elif self.activation == 'sigmoid':
                initializer = 'xavier_unif'
            elif self.activation == 'tanh':
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
        
        self.params = [self.W, self.b]
            

class Dropout(Layer):
    def __init__(self, drop_rate=.5, correct_magnitude=True):
        super().__init__()
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
        return (dA,)

    def _initialize(self, in_dim):
        self.input_dim = in_dim
        self.output_dim = in_dim

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward_propagate(self, X, inference=False):
        return X.reshape((len(X), -1))

    def backward_propagate(self, dA):
        return (dA, )
    
    def _initialize(self, in_dim):
        self.input_dim = in_dim
        out_dim = 1
        for dim in in_dim:
            out_dim *= dim
        self.output_dim = out_dim
    
class BatchNorm(Layer):
    def __init__(self, beta=.99, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.gammas = None
        self.betas = None
        self.mus = None
        self.sigmas = None
    
    def forward_propagate(self, X, inference=False):
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        sigma = np.where(sigma >= self.eps, sigma, self.eps)
        if inference:
            Z = (X - self.mus) / (self.sigmas + self.eps)
        else:
            self.last_input = X
            self.mus = self.beta * self.mus + (1 - self.beta) * mu
            self.sigmas = self.beta * self.sigmas + (1 - self.beta) * sigma
            Z = (X - mu) / sigma
        A = Z * self.gammas + self.betas
        
        if not inference:
            self.last_output = A
            self.last_mu = mu
            self.last_sigma = sigma
            self.input_centered = X - mu
        return A
    
    def backward_propagate_legasy(self, dy):
        dy = dy.T

        n, d = dy.shape

        dgamma = np.sum(self.last_output * dy, axis=0, keepdims=True)
        dbeta = np.sum(dy, axis=0, keepdims=True)

        dZ = dy * self.gammas
        inv_var = 1 / self.last_sigma
        Xhat = self.last_output
        dx = (1. / n) * inv_var * (n * dZ - np.sum(dZ, axis=0) - Xhat * np.sum(dZ * Xhat, axis=0))
        dA_prev = dx
        return dA_prev.T, dgamma, dbeta
    
    
    def backward_propagate(self, dy):
        dy = dy.T

        n, d = dy.shape

        dgamma = np.sum(self.last_output * dy, axis=0, keepdims=True)
        dbeta = np.sum(dy, axis=0, keepdims=True)

        dZ = dy * self.gammas
#         inv_var = 1 / np.sqrt(self.last_sigma ** 2 + self.eps)
        Xhat = self.last_output

        dvar = np.sum(dZ * (self.last_input - self.last_mu) * (-.5 * (self.last_sigma**2 + self.eps) ** (-3/2)), axis=0)
    
        p1 = np.sum(-dZ / self.last_sigma, axis=0)
        p2 = dvar * (-2/n) * np.sum(self.last_input - self.last_mu, axis=0)
        dmu = p1 + p2
        o1 = dZ / self.last_sigma
        o2 =  dvar * (2/n) * (self.input_centered)
        o3 = dmu / n
        dx = o1 + o2 + o3

        dA_prev = dx
        return dA_prev.T, dgamma, dbeta
    
    def _initialize(self, in_dim):
        self.input_dim = in_dim
        self.output_dim = in_dim
        if type(in_dim) is int:
            shape = (1, in_dim)
        else:
            shape = (1, *in_dim[1:])
        self.gammas = np.ones(shape)
        self.betas = np.zeros(shape)

        self.sigmas = np.ones(shape)
        self.mus = np.zeros(shape)

        self.params = [self.gammas, self.betas]

    @property
    def _n_params(self):
        # 4x the number of activations of the previous layer
        return self.gammas.size + self.betas.size + self.sigmas.size + self.mus.size