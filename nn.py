import numpy as np
from loss import *
from layers import *
from activations import *

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
    def __init__(self, n_iterations=100, learning_rate=0.05, loss=MSE(), verbose=True, verbose_step=100):
        """
        Class containing Neural Network architecture: Layers and Optimizer
        
        @param learning_rate: scale factor for parameters gradient update
        @param loss: loss function to optimize
        @param verbose: if True, learning process will output some statistics
        """
        self.layers = []
        # Optimizer stuff
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.loss = loss
        self.verbose = verbose
        self.verbose_step = verbose_step
    
    def fit(self, X, Y):
        for i in range(self.n_iterations):
            y_pred = self.forward_(X)
            cost = self.loss.forward(Y, y_pred)
            if (self.verbose and i % self.verbose_step == 0):
                print(f"{i}: {self.loss}={cost}")
            self.backward_(Y, y_pred)
    
    def forward_(self, X, inference=False):
        """
        Propagate input through consecutive layers
        """
        for l in self.layers:
            X = l.forward_propagate(X, inference=inference)
        return X
    
    def backward_(self, Y, Y_pred):
        dA = self.loss.backward(Y, Y_pred).T
        grads = {}
        for i, l in enumerate(reversed(self.layers)):
            dA, dW, db = l.backward_propagate(dA)
            grads[l] = dW, db
        
        # Optimization step
        for i, l in enumerate(reversed(self.layers)):
            if l.trainable:
                dW, db = grads[l]
                l.W = l.W - self.learning_rate * dW.T
                l.b = l.b - self.learning_rate * db.T      
    
    def predict(self, X):
        return self.forward_(X, inference=True)
    
    def add(self, layer):
        self.layers.append(layer)
    
    def summary(self):
        return '\n'.join([repr(l) for l in self.layers])

if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add(Layer(1, 2, activation=Sigmoid()))
    nn.add(Layer(2, 1, activation=Linear()))
    print(nn.summary())