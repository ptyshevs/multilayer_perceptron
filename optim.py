import numpy as np

class LRScheduler:
    def __init__(self):
        pass
    
    def __call__(self, t):
        pass

class ConstantLR(LRScheduler):
    def __init__(self, lr=0.05):
        self.lr = lr
    
    def __call__(self, t=None):
        return self.lr

class AnnealingLR(LRScheduler):
    def __init__(self, lr=0.05, a=.99996):
        self.lr = lr
        self.a = a
    
    def __call__(self, t=0:
        self.lr = self.lr * self.a
        return self.lr

class CircluarLR(LRScheduler):
    def __init__(self, lr=.05, scale=.025, periodic_fn=np.cos):
        self.lr = lr
        self.scale = scale
        self.periodic_fn = periodic_fn
    
    def __call__(self, t=0):
        return np.clip(self.lr + self.periodic_fn(t) * self.scale, 0, 1)

lr_map = {'const': ConstantLR,
          'anneal': AnnealingLR,
          'circular': CircluarLR}    

def learning_rate_mapper(lr):
    if type(lr) is float:
        return ConstantLR(lr)
    elif type(lr) is str:
        if lr.lower() in lr_map:
            return lr_map[lr.lower()]()
        else:
            raise ValueError(f"'{lr}' is not recognized")
    elif issubclass(type(lr), LRScheduler):
        return lr
    else:
        return ConstantLR(lr)
    
class Optimizer:
    def __init__(self):
        """ Class responsible for training process """
        pass
    
class GradientDescent(Optimizer):
    def __init__(self, learning_rate=.05):
        self.learning_rate = learning_rate_mapper(learning_rate)
    
    def __call__(self, layer, params_grad, **kwargs):
        for i in range(len(layer.params)):
            # Don't modify: this is in-place operation
            layer.params[i] -= self.learning_rate(kwargs['t']) * params_grad[i]

class MomentumGradientDescent(Optimizer):
    def __init__(self, learning_rate=.05, beta=.9, bias_correction=True):
        self.learning_rate = learning_rate_mapper(learning_rate)
        self.beta = beta
        self.bias_correction = bias_correction
        self.moments = dict()
    
    def __call__(self, layer, params_grad, **kwargs):
        if layer not in self.moments:
            self.moments[layer] = []
            for param in layer.params:
                self.moments[layer].append(np.zeros_like(param))
        
        for i in range(len(layer.params)):
            self.moments[layer][i] = self.beta * self.moments[layer][i] + (1 - self.beta) * params_grad[i]
            layer.params[i] -= self.learning_rate(kwargs['t']) * self.moments[layer][i]
            if self.bias_correction:
                layer.params[i] /= (1 - self.beta ** (1 + kwargs['t']))
        
    
optimizers_map = {'gd': GradientDescent(), 'sgd': GradientDescent(),
                  'momentum_gd': MomentumGradientDescent()}

def optimizer_mapper(optimizer):
    if type(optimizer) is str:
        if optimizer.lower() in optimizers_map:
            return optimizers_map[optimizer.lower()]
        else:
            raise ValueError(f"Optimizer '{optimizer}' is not recognized")
    else:
        return optimizer