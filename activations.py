import numpy as np

class Activation:
    def __init__(self):
        """
        Base class for activations
        """
        pass
    
    def forward(self, X):
        raise NotImplementedError("This is base class")
    
    def backward(self, Z):
        raise NotImplementedError("This is base class")
    
    def __repr__(self):
        return 'activation (base class)'
    
    def __eq__(self, other):
        return repr(self) == repr(other)

class Identity(Activation):
    
    def forward(self, X):
        return X
    
    def backward(self, Z):
        return Z
    
    def __repr__(self):
        return 'identity'
    
class Linear(Activation):
    def __init__(self):
        self.last_input = None
    
    def forward(self, X):
        self.last_input = X
        return X
    
    def backward(self, Z):
        s = self.last_input
        return s * Z
    
    def __repr__(self):
        return 'linear'

class Sigmoid(Activation):
    def __init__(self, protected=True):
        self.last_input = None
        self.bias = 1e-10 if protected else 0
    
    def forward(self, X):
        self.last_input = X
        return self._sigmoid(X)
    
    def backward(self, Z):
        s = self._sigmoid(Z)
        return s * (1 - s)
    
    def _sigmoid(self, X):
        return 1 / ((1 + np.exp(-X)) + self.bias)
    
    def __repr__(self):
        return 'sigmoid'

class Tanh(Activation):
    def __init__(self):
        pass
    
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, Z):
        return 1 - np.tanh(Z) ** 2
    
    def __repr__(self):
        return 'tanh'

class Relu(Activation):
    
    def forward(self, X):
        return np.where(X >= 0, X, 0.0)
    
    def backward(self, Z):
        return np.where(Z >= 0, 1.0, 0.0)
    
    def __repr__(self):
        return 'relu'

class LeakyRelu(Activation):
    def __init__(self, alpha=.3):
        self.alpha = alpha

    def forward(self, X):
        return np.where(X >= 0, X, self.alpha * X)
    
    def backward(self, Z):
        return np.where(Z >= 0, 1.0, self.alpha)
    
    def __repr__(self):
        return 'leaky_relu'

# class PRelu(Activation):
#     def __init__(self):
#         """
#         LeakyRelu, but coef. with x < 0 learned from data
#         """
#         pass
    
#     def update(self):

class ELU(Activation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, X):
        return np.where(X >= 0, X, self.alpha * (np.exp(X) - 1))
    
    def backward(self, Z):
        return np.where(Z >= 0, 1.0, self.alpha * np.exp(Z))
    
    def __repr__(self):
        return 'elu'

        

class Quadratic(Activation):
    
    def forward(self, X):
        return X ** 2
    
    def backward(self, Z):
        return 2 * Z
    
    def __repr__(self):
        return 'quad'
    

class Softmax(Activation):
    def __init__(self):
        """
        Softmax is an activation function that transforms
        vector of real numbers into probability distribution
        """
        pass
  
    def forward(self, X):
        return self._softmax(X)
  
    def backward(self, Z):
        """
        Full matrix of partial derivatives is nxn.
        Here I leave only diagonal vector of a Jacobian (for each z_i calculate dz_i\dx_i)
        """
#         logits = self.forward(Z)
#         n, l = logits.shape
        
#         Jac = np.zeros((n, l, l))
#         for k in range(n):
#             for i in range(l):
#                 for j in range(l):
#                     if i == j:
#                         Jac[k, i,j] = logits[k, i] * (1 - logits[k, i])
#                     else:
#                         Jac[k, i,j] = - logits[k, i] * logits[k, j]
#         return Jac
        s = self._softmax(Z)
        return s * (1 - s)
    
    def _softmax(self, v, scale=True):
        """
        Scaling is used for better numerical stability
        """
        if scale:
            v -= np.max(v)
        return (np.exp(v).T / np.exp(v).sum(axis=1)).T
    
    def __repr__(self):
        return 'softmax'
    
activations = {'identity':Identity,
               'linear': Linear,
                'sigmoid': Sigmoid,
               'tanh': Tanh,
               'softmax': Softmax,
               'relu': Relu,
               'leaky_relu': LeakyRelu,
               'elu': ELU,
               'quadratic': Quadratic}

def activation_to_obj(name):
    if name in activations:
        return activations[name]()
    else:
        raise ValueError(f"Activation is not recognized: {name}")