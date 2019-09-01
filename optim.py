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
    
    def __call__(self, t=0):
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
            self.moments[layer] = [np.zeros_like(p) for p in payer.params]
        t = kwargs['t']
        grad_scale = self.learning_rate(t)
        if self.bias_correction:
            grad_scale /= (1 - self.beta ** (1 + t))
        for i in range(len(layer.params)):
            self.moments[layer][i] = self.beta * self.moments[layer][i] + (1 - self.beta) * params_grad[i]
            # w = w - a*v
            # b = b - a*v

            layer.params[i] -= grad_scale * self.moments[layer][i]

class RMSProp(Optimizer):
    def __init__(self, lr=.05, beta=.9, eps=1e-9):
        self.lr = learning_rate_mapper(lr)
        self.beta = beta
        self.moments = dict()
        self.eps = eps
    
    def __call__(self, layer, params_grad, **kwargs):
        if layer not in self.moments:
            self.moments[layer] = [np.zeros_like(p) for p in layer.params]
            
        t = kwargs['t']
        lr = self.lr(t)
        for i, p in enumerate(layer.params):
            self.moments[layer][i] *= self.beta
            self.moments[layer][i] += (1 - self.beta) * params_grad[i] ** 2
            
            grad_scale = lr / np.sqrt(self.moments[layer][i] + self.eps)
            
            layer.params[i] -= grad_scale * params_grad[i]
            
class Adam(Optimizer):
    def __init__(self, lr=.05, beta1=.9, beta2=.999, eps=1e-9):
        self.lr = learning_rate_mapper(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.moments = dict()
        self.norms = dict()
    
    def __call__(self, layer, params_grad, **kwargs):
        if layer not in self.moments:
            self.moments[layer] = [np.zeros_like(p) for p in layer.params]
            self.norms[layer] = [np.zeros_like(p) for p in layer.params]
        
        t = kwargs['t']
        lr = self.lr(t)
        for i, p in enumerate(layer.params):
            self.moments[layer][i] *= self.beta1
            self.moments[layer][i] += (1 - self.beta1) * params_grad[i]
            
            self.norms[layer][i] *= self.beta2
            self.norms[layer][i] += (1 - self.beta2) * params_grad[i] ** 2
            
            # Bias correction on both
            
            moment_corrected = self.moments[layer][i] / (1 - self.beta1 ** (t+1))
            norm_corrected = self.norms[layer][i] / (1 - self.beta2 ** (t+1))
            
            layer.params[i] -= lr * moment_corrected / np.sqrt(norm_corrected + self.eps)
            
            
            
optimizers_map = {'gd': GradientDescent, 'sgd': GradientDescent,
                  'momentum_gd': MomentumGradientDescent,
                  'rmsprop': RMSProp, 'adam': Adam}

def optimizer_mapper(optimizer):
    if type(optimizer) is str:
        if optimizer.lower() in optimizers_map:
            return optimizers_map[optimizer.lower()]()
        else:
            raise ValueError(f"Optimizer '{optimizer}' is not recognized")
    else:
        return optimizer