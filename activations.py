import numpy as np

class Softmax:
    def __init__(self):
      """ Softmax is an activation function that transforms
      vector of real numbers into probability distribution """
      pass
  
    def forward(self, X):
        return self._softmax(X)
  
    def backward(self, Z):
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
        return self._softmax(Z) * (1 - self._softmax(Z))
    
    def _softmax(self, v, scale=True):
        if scale:
            v -= np.max(v)
        return (np.exp(v).T / np.exp(v).sum(axis=1)).T

class Linear:
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

class Sigmoid:
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

class Tanh:
    def __init__(self):
        pass
    
    def forward(self, X):
        return np.tanh(X)
    
    def backward(self, Z):
        return 1 - np.tanh(Z) ** 2

class Relu:
    
    def forward(self, X):
        return np.where(X >= 0, X, 0.0)
    
    def backward(self, Z):
        return np.where(Z >= 0, 1.0, 0.0)
    
    def __repr__(self):
        return 'relu'

class Quadratic:
    
    def forward(self, X):
        return X ** 2
    
    def backward(self, Z):
        return 2 * Z
    
    def __repr__(self):
        return 'quad'