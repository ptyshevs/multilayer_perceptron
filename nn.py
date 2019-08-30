import numpy as np
from loss import *
from layers import *
from activations import *
from metrics import metric_mapper
import pickle

class History:
    def __init__(self):
        """
        History object contains all the necessary information about the training process
        """
        self.history = dict()
    
    def add(self, entry):
        for (k, v) in entry.items():
            if k in self.history:
                self.history[k].append(v)
            else:
                self.history[k] = [v]
    
    def get(self, idx):
        return self.history[idx]
    
    def __iter__(self):
        return iter(self.history)

    def __getitem__(self, k):
        return self.history[k]
    
    @property
    def keys(self):
        return self.history.keys()
    
    def __repr__(self):
        return 'History:' + '|'.join([str(k) for k in self.keys])

class Optimizer:
    def __init__(self):
        """ Class responsible for training process """
        pass

class EarlyStopping:
    def __init__(self, patience=6, monitor='loss'):
        self.patience = patience
        self.cnt = 0
        self.prev_val = None
        self.monitor = monitor
        
    def restart(self):
        self.prev_val = None
        self.cnt = 0
    
    def __call__(self, history, model):
        last_entry = history[-1]
        val = last_entry[self.monitor]
        has_improved = False
        if self.monitor.endswith('loss'):
            if self.prev_val is None or self.prev_val > val:
                has_improved = True
        else:
            if self.prev_val is None or self.prev_val < val:
                has_improved = True
        
        if has_improved:
            self.prev_val = val
            self.cnt = 0
        else:
            self.cnt += 1
        
        stop = False
        if self.cnt >= self.patience:
            model.should_stop = True
    
class GradientDescent:
    def __init__(self, learning_rate=.01):
        self.learning_rate = learning_rate
    
    def update(self, theta, grad):
        return theta - self.learning_rate * grad

class NeuralNetwork:
    def __init__(self, loss=None, learning_rate=0.05, verbose=False, verbose_step=100, debug=False):
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
        self.should_stop = False
        self.initialized = False
        self.n_epochs = 0  # I need this to be able to control training outside .fit()
    
    def fit(self, X, Y, X_val=None, Y_val=None, n_epochs=1, callbacks=None, metrics=[], reinitialize=True):
        if not self.initialized or reinitialize:
            self._initialize(X.shape[1])

        history = History()
        validation_provided = X_val is not None and Y_val is not None

        for i in range(1, max(n_epochs, self.n_epochs) + 1):
            history_entry = {"epoch": i}

            if validation_provided:
                history_entry['val_loss'] = self.loss.forward(Y_val, self.forward_(X_val))
            
                        
            y_pred = self.forward_(X)
            cost = self.loss.forward(Y, y_pred)
            
            history_entry["loss"] = cost
            
            for metric in metrics:
                metric_obj = metric_mapper(metric)
                history_entry[metric_obj.__name__] = metric_obj(Y, y_pred)
                if validation_provided:
                    history_entry['val_' + metric_obj.__name__] = metric_obj(Y_val, self.forward_(X_val, inference=True))

            
            if (self.verbose and (i % self.verbose_step == 0 or i == n_epochs)) or self.debug:
                self._handle_output(history_entry, n_epochs)
            
            self.backward_(Y, y_pred)
            
            if callbacks:
                for cb in callbacks:
                    cb(Y, y_pred)
            
            history.add(history_entry)
            
            if self.should_stop:
                break

        return history
    
    def _handle_output(self, entry, n_epochs):
        print(f"[{entry['epoch']}/{n_epochs}]: ", end='')
        print(f"loss={entry['loss']:.5f}", end=' ')
        if 'val_loss' in entry:
            print(f'val_loss={entry["val_loss"]:.5f}', end=' ')
        for k, v in entry.items():
            if k in ['epoch', 'loss', 'val_loss']:
                continue
            print(f"{k}={v:.5f}", end=' ')
        print('')
    
    def _initialize(self, in_dim):
        if self.layers:
            self.layers[0]._initialize(in_dim)
        for i in range(1, len(self.layers)):
            self.layers[i]._initialize(self.layers[i-1].output_dim)
        
        self.initialized = True
            
    
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
        if not self.initialized:
            raise ValueError("NNet is not trained.")
        return self.forward_(X, inference=True)
    
    def add(self, layer):
        self.layers.append(layer)
    
    def summary(self):
        trainable_params = 0
        total_params = 0
        for l in self.layers:
            print(repr(l))
            total_params += l._n_params
            if l.trainable:
                trainable_params += l._n_params
        print("=======================================================")
        print('Total number of parameters:\t\t', total_params)
        print("Total number of trainable params:\t", trainable_params)
    
    def _loss_mapper(self, loss):
        if type(loss) is str:
            return loss_to_obj(loss)
        else:
            return loss
        

if __name__ == '__main__':
    nn = NeuralNetwork('crossentropy')
    nn.add(Layer(2, activation='sigmoid'))
    nn.add(Layer(1, activation='linear'))
    nn.summary()