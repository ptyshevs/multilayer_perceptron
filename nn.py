import numpy as np

def random_initializer(n, m, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.randn(n, m) * .01

def zero_initializer(n, m):
    return np.zeros((n, m))

class Linear:
    def __init__(self):
        self.last_input = None
    
    def forward(self, X):
        self.last_input = X
        return X
    
    def backward(self, dY):
        s = self.last_input
        return s * dy
    
    def __repr__(self):
        return 'linear'

class Sigmoid:
    def __init__(self):
        self.last_input = None
    
    def forward(self, X):
        self.last_input = X
        return self._sigmoid(X)
    
    def backward(self, dY):
        s = self._sigmoid(self.last_input)
        return s * (1 - s) * dY
    
    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def __repr__(self):
        return 'sigmoid'

# class Linear:
#     def __init__(self, W, b):
#         self.last_input = None

    
#     def forward(self, X):
#         self.last_input = X
#         return self._sigmoid(X)
    
#     def backward(self, dY):
#         s = self._sigmoid(self.last_input)
#         return s * (1 - s) * dY
        

class Layer:
    def __init__(self, input_dim, output_dim, activation=None):
        """ Linear -> Activation dense layer """
        self.input_dim, self.output_dim = input_dim, output_dim
        self.activation = activation
        self.last_input = None
        
        self.W = random_initializer(output_dim, input_dim)
        self.b = zero_initializer(output_dim, 1)
    
    def forward_propagate(self, X):
        self.last_input = X  # Cache last input
        
        Z = self.W @ X + self.b
        self.last_output = Z
        if self.activation:
            return self.activation.forward(Z)
        else:
            return Z

    def backward_propagate(self, dY):
        
        dZ = self.activation.backward(dY)
        dW = self.last_input.T @ dZ
        db = np.mean(dZ, axis=1, keepdims=True)
        return dW, db
    
    def __repr__(self):
        return f"{self.activation} # params = {self._n_params()}"
    
    def _n_params(self):
        w = self.W.shape[0] * self.W.shape[1]
        b = self.b.shape[0]
        return w + b

class Optimizer:
    def __init__(self):
        """ Class responsible for training process """
        pass

class GradientDescent:
    def __init__(self, learning_rate=.01):
        self.learning_rate = learning_rate
    
    def update(self, theta, grad):
        return theta - self.learning_rate * grad

class NeuralNetwork:
    def __init__(self):
        """ Class containing Neural Network architecture: Layers and Optimizer """
        self.layers = []
    
    def fit(self, X, Y):
        pass
    
    def predict(self, X):
        pass
    
    def add(self, layer):
        self.layers.append(layer)
    
    def summary(self):
        return '\n'.join([repr(l) for l in self.layers])

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(Layer(1, 2, activation=Sigmoid()))
    nn.add(Layer(2, 1, activation=Linear()))
    print(nn.summary())