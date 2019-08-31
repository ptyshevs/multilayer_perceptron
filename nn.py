import numpy as np
from loss import *
from layers import *
from activations import *
from metrics import metric_mapper
from optim import optimizer_mapper
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


class NeuralNetwork:
    def __init__(self, loss=None, optimizer=None, verbose=False, verbose_step=100, debug=False):
        """
        Class containing Neural Network architecture: Layers and Optimizer
        
        @param learning_rate: scale factor for parameters gradient update
        @param loss: loss function to optimize
        @param optimizer: heurisitic for gradient descent
        @param verbose: if True, learning process will output some statistics
        """
        self.layers = []
        self.loss = loss_mapper(loss)
        self.optim = optimizer_mapper(optimizer)
        self.verbose = verbose
        self.verbose_step = verbose_step
        self.debug = debug

        self.should_stop = False
        self.initialized = False

        self.n_epochs = 0
    
    def _pre_fit(self, train_params):
        self.should_stop = False
        n_samples, n_features = train_params['X'].shape
        if train_params['X_val'] is not None:
            val_features = train_params['X_val'].shape[1]
            assert n_features == val_features
        
        if self.loss is None:
            print("Loss is not specified")
            self.should_stop = True

        if self.optim is None:
            print("Optimizer is not specified")
            self.should_stop = True
        
        if not self.initialized or train_params['reinitialize']:
            self._initialize(n_features)
        
        if train_params['callbacks']:
            for cb in train_params['callbacks']:
                cb.restart()
        
        if train_params['batch_size'] == 0:
            train_params['batch_size'] = n_samples
        
        train_params['val'] = train_params['X_val'] is not None and train_params['Y_val'] is not None
        
        mapped_metrics = []
        for m in train_params['metrics']:
            mapped_metrics.append(metric_mapper(m))
        train_params['metrics'] = mapped_metrics

        
    def _record_history_entry(self, params):
        entry = {"epoch": params['epoch']}
        
        X, Y, X_val, Y_val = params['X'], params['Y'], params['X_val'], params['Y_val']
        
        Y_pred = self.forward_(X, inference=True)
        Y_val_pred = None
        if params['val']:
            Y_val_pred = self.forward_(X_val, inference=True)
        
        entry['loss'] = self.loss.forward(Y, Y_pred)
        if params['val']:
            entry['val_loss'] = self.loss.forward(Y_val, Y_val_pred)
        
        for metric in params['metrics']:
            entry[metric.__name__] = metric(Y, Y_pred)
            if params['val']:
                entry['val_' + metric.__name__] = metric(Y_val, Y_val_pred)
        return entry
        
    def _on_epoch_end(self, params):
        entry = self._record_history_entry(params)
        
        if params['callbacks']:
            for cb in params['callbacks']:
                cb(entry, self)

        params['history'].add(entry)
        
        epoch = entry['epoch']
        n_epochs = params['n_epochs']
        
        if (self.verbose and (epoch % self.verbose_step == 0 or epoch == n_epochs)) or self.debug:
            self._handle_output(entry, n_epochs)


    def fit(self, X, Y, X_val=None, Y_val=None, n_epochs=1, batch_size=0, 
            callbacks=None, metrics=[], reinitialize=True):

        params = {'batch_size': batch_size, 'callbacks': callbacks, 'reinitialize': reinitialize,
                        'metrics': metrics, 'callbacks': callbacks, 'history': History(),
                        'X': X, 'Y': Y, 'X_val': X_val, 'Y_val': Y_val, 'n_epochs': n_epochs}

        self._pre_fit(params)
        
        batch_size = params['batch_size']
        metrics = params['metrics']

        n_steps = len(X) // batch_size
        if len(X) % batch_size != 0:  # There is a last batch that is not full
            n_steps += 1

        for i in range(1, max(n_epochs, self.n_epochs) + 1):
            if self.should_stop:
                break
            start_idx, end_idx = 0, batch_size
            params['epoch'] = i
            for step in range(n_steps):
                
                X_batch = X[start_idx:end_idx, :]
                Y_batch = Y[start_idx:end_idx, :]
                
                y_pred = self.forward_(X_batch)

                grads = self.backward_(Y_batch, y_pred)
                self._optimize(grads)
                
                start_idx += batch_size
                end_idx += batch_size
            self._on_epoch_end(params)

        return params['history']
    
    def forward_(self, X, inference=False):
        """
        Propagate input through consecutive layers
        """
        for l in self.layers:
            X = l.forward_propagate(X, inference=inference)
        return X
    
    def backward_(self, Y, Y_pred):
        dA = self.loss.backward(Y, Y_pred).T
        grads = dict()
        for i, l in enumerate(reversed(self.layers)):
            dA, *dParams = l.backward_propagate(dA)
            grads[l] = dParams
        return grads
    
    
    def _optimize(self, grads):
        for i, l in enumerate(reversed(self.layers)):
            if not l.trainable:
                continue
            dW, db = grads[l]
            l.W = self.optim(l.W, dW.T)
            l.b = self.optim(l.b, db.T)
    
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
        
    def _handle_output(self, entry, n_epochs):
        print(f"[{entry['epoch']:5}/{n_epochs}]: ", end='')
        for k, v in entry.items():
            if k == 'epoch':
                continue
            print(f"{k}={v:.5f}", end=' ')
        print('')
    
    def _initialize(self, in_dim):
        if self.layers:
            self.layers[0]._initialize(in_dim)
        for i in range(1, len(self.layers)):
            self.layers[i]._initialize(self.layers[i-1].output_dim)
        
        self.initialized = True
        
        

if __name__ == '__main__':
    nn = NeuralNetwork('crossentropy')
    nn.add(Layer(2, activation='sigmoid'))
    nn.add(Layer(1, activation='linear'))
    nn.summary()