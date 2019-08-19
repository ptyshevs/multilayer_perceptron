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
    def __init__(self, loss, learning_rate=0.05, verbose=False, verbose_step=100, debug=False):
        """
        Class containing Neural Network architecture: Layers and Optimizer
        
        @param learning_rate: scale factor for parameters gradient update
        @param loss: loss function to optimize
        @param verbose: if True, learning process will output some statistics
        """
        self.layers = []
        # Optimizer stuff
        self.learning_rate = learning_rate
        self.loss = self._loss_mapper(loss)
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.debug = debug
    
    def fit(self, X, Y, n_epochs=1):
        history = []
        for i in range(n_epochs):
            y_pred = self.forward_(X)
            if self.debug:
                print("y_pred:", y_pred)
            cost = self.loss.forward(Y, y_pred)
            if (self.verbose and i % self.verbose_step == 0) or self.debug:
                print(f"{i}: {self.loss}={cost}")
            self.backward_(Y, y_pred)
        return history
    
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
    
    def _loss_mapper(self, loss):
        if type(loss) is str:
            return loss_to_obj(loss)
        else:
            return loss
        

if __name__ == '__main__':
    nn = NeuralNetwork('crossentropy')
    nn.add(Layer(1, 2, activation='sigmoid'))
    nn.add(Layer(2, 1, activation='linear'))
    print(nn.summary())