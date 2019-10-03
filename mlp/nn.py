import numpy as np
from .loss import *
from .layers import *
from .activations import *
from .metrics import metric_mapper
from .optim import *
import tqdm

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
        return 'History: ' + '|'.join([str(k) for k in self.keys])


class NeuralNetwork:
    def __init__(self, loss=None, optimizer=None, verbose=False, verbose_step=100, debug=False,
                 random_state=None):
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
        self.random_state = random_state

        self.should_stop = False
        self.initialized = False

        self.n_epochs = 0
    
    def _pre_fit(self, params):
        self.should_stop = False
        
        X, Y, X_val, Y_val = params['X'], params['Y'], params['X_val'], params['Y_val']
        params['val'] = params['X_val'] is not None and params['Y_val'] is not None

        if len(X.shape) < 2:
            X = X.reshape(-1, 1)
        if len(Y.shape) < 2:
            Y = Y.reshape(-1, 1)
        if params['val']:
            if len(X_val.shape) < 2:
                X_val = X_val.reshape(-1, 1)
            if len(Y_val.shape) < 2:
                Y_val = Y_val.reshape(-1, 1)
        
        params['X'] = X
        params['Y'] = Y
        params['X_val'] = X_val
        params['Y_val'] = Y_val
        
        n_samples, *n_features = X.shape
        if X_val is not None:
            val_samples, *val_features = X_val.shape
            assert n_features == val_features

        if len(n_features) == 1:
            n_features = n_features[0]

        if self.loss is None:
            print("Loss is not specified")
            self.should_stop = True

        if self.optim is None:
            print("Optimizer is not specified")
            self.should_stop = True
        
        if not self.initialized or params['reinitialize']:
            self._initialize(n_features)
        
        if params['callbacks']:
            for cb in params['callbacks']:
                cb.restart()
        
        if params['batch_size'] == 0:
            params['batch_size'] = n_samples
        
        mapped_metrics = []
        for m in params['metrics']:
            mapped_metrics.append(metric_mapper(m))
        params['metrics'] = mapped_metrics
    
    def compile(self, input_features):
        """
        Compile model to enforce initialization of the parameters
        """
        self._initialize(input_features)
    
    def get_params(self):
        """
        Returns list of references to parameters of individual layers
        """
        return [l.params for l in self.layers]
    
    def set_params(self, params):
        """
        Set parameter values for each layer. Same format is expected as in get_params().
        """
        for layer, params in zip(self.layers, params):
            layer.params = params

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
        if self.random_state is not None:
            np.random.seed(self.random_state)

        params = {'batch_size': batch_size, 'callbacks': callbacks, 'reinitialize': reinitialize,
                        'metrics': metrics, 'callbacks': callbacks, 'history': History(),
                        'X': X, 'Y': Y, 'X_val': X_val, 'Y_val': Y_val, 'n_epochs': n_epochs}

        self._pre_fit(params)
        X, Y, X_val, Y_val = params['X'], params['Y'], params['X_val'], params['Y_val']
        batch_size = params['batch_size']
        metrics = params['metrics']

        n_steps = len(X) // batch_size
        if len(X) % batch_size != 0:  # There is a last batch that is not full
            n_steps += 1
        
        t = 0
        for i in range(1, max(n_epochs, self.n_epochs) + 1):
            if self.should_stop:
                break
            start_idx, end_idx = 0, batch_size
            params['epoch'] = i
            for step in tqdm.tnrange(n_steps):
                if self.should_stop:
                    break
                X_batch = X[start_idx:end_idx, ...]
                Y_batch = Y[start_idx:end_idx, ...]
                
                y_pred = self.forward_(X_batch)

                grads = self.backward_(Y_batch, y_pred)
                self._optimize(grads, t, batch_size)
                
                start_idx += batch_size
                end_idx += batch_size
                
                t += 1
                # loss = self.loss.forward(Y_batch, y_pred)
                # if self.debug and np.isnan(loss):
                #     print(f'= loss is nan on epoch {i} | step {step}',)
                #     self.should_stop = True
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

    def _optimize(self, grads, t, batch_size):
        for i, l in enumerate(reversed(self.layers)):
            if not l.trainable:
                continue
            # Don't remove t from function call - it is used for learning rate scheduling
            self.optim(l, grads[l], t=t, batch_size=batch_size)
    
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
        self.optim.reset()


if __name__ == '__main__':
    nn = NeuralNetwork('crossentropy')
    nn.add(Layer(2, activation='sigmoid'))
    nn.add(Layer(1, activation='linear'))
    nn.summary()
