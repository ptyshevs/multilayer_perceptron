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
        s = self._sigmoid(dY)
        return s * (1 - s)
    
    def _sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def __repr__(self):
        return 'sigmoid'

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
#         print(f"W.shape={self.W.shape} | X.shape={X.shape}")
        Z = X @ self.W.T + self.b.T
        self.last_output = Z
        if self.activation:
            return self.activation.forward(Z)
        else:
            return Z

    def backward_propagate(self, dA):
        if self.activation is not None:
            dZ = self.activation.backward(self.last_output) * dA.T
        else:
            dZ = dA.T
#         print(f"last_input_shape={self.last_input.shape}, dZ.shape={dZ.shape}")
        dW = self.last_input.T @ dZ / len(dZ)
        db = np.mean(dZ, axis=0, keepdims=True)
        dA_prev = (self.W.T @ dZ.T)
        return dA_prev, dW, db
    
    def __repr__(self):
        return f"{self.activation} # params = {self._n_params()}"
    
    def _n_params(self):
        w = self.W.shape[0] * self.W.shape[1]
        b = self.b.shape[0]
        return w + b

class MSE:
    def forward(self, Y, Y_pred):
        return np.sum(np.power(Y - Y_pred, 2))
    
    def backward(self, Y, Y_pred):
        return -2 * (Y - Y_pred) / Y.shape[1]

    def __repr__(self):
        return 'MSE'

class Binary:
    def forward(self, Y, Y_pred):
        return -(Y * np.log(np.clip(Y_pred, 1e-12, None)) + (1 - Y) * np.log(np.clip(1 - Y_pred, 1e-12, None))).mean()
    
    def backward(self, Y, Y_pred):
        return - (np.divide(Y, Y_pred) - np.divide(1 - Y, 1 - Y_pred))
    
    def __repr__(self):
        return "Binary cross-entropy"

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
    def __init__(self, n_iterations=100, learning_rate=0.01, loss=MSE(), verbose=True):
        """ Class containing Neural Network architecture: Layers and Optimizer """
        self.layers = []
        # Optimizer stuff
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss
        self.verbose = verbose
    
    def fit(self, X, Y):
        for i in range(self.n_iterations):
            y_pred = self.forward_(X)
            cost = self.loss.forward(Y, y_pred)
            if (self.verbose and i % 1 == 0):
                print(f"{i}: {self.loss}={cost}")
            self.backward_(Y, y_pred)
    
    def forward_(self, X):
        for l in self.layers:
            X = l.forward_propagate(X)
        return X
    
    def backward_(self, Y, Y_pred):
        dA = self.loss.backward(Y, Y_pred).T
        grads = {}
        for i, l in enumerate(reversed(self.layers)):
            dA, dW, db = l.backward_propagate(dA)
            grads[l] = dW, db
        for i, l in enumerate(reversed(self.layers)):
            dW, db = grads[l]
            l.W = l.W - self.learning_rate * dW.T
            l.b = l.b - self.learning_rate * db.T      
    
    def predict(self, X):
        return self.forward_(X)
    
    def add(self, layer):
        self.layers.append(layer)
    
    def summary(self):
        return '\n'.join([repr(l) for l in self.layers])

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(Layer(1, 2, activation=Sigmoid()))
    nn.add(Layer(2, 1, activation=Linear()))
    print(nn.summary())