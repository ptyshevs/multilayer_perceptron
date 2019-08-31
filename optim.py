import numpy as np

class LRScheduler:
    def __init__(self):
        pass
    
    def __call__(self, t):
        pass

class ConstantLR(LRScheduler):
    def __init__(self, lr):
        self.lr = lr
    
    def __call__(self, t):
        return self.lr

class AnnealingLR(LRScheduler):
    def __init__(self, lr=0.05, a=.9996):
        self.lr = lr
        self.a = a
    
    def __call__(self, t):
        self.lr = self.lr * self.a
        return self.lr

class CircluarLR(LRScheduler):
    def __init__(self, lr=.05, scale=.025, periodic_fn=np.sin):
        self.lr = lr
        self.scale = scale
        self.periodic_fn = periodic_fn
    
    def __call__(self, t):
        return np.clip(self.lr + self.periodic_fn(t) * self.scale, 0, 1)

class Optimizer:
    def __init__(self):
        """ Class responsible for training process """
        pass
    
class GradientDescent:
    def __init__(self, learning_rate=.05):
        self.learning_rate = learning_rate
    
    def __call__(self, params, params_grad):
        return params - self.learning_rate * params_grad

optimizers_map = {'gd': GradientDescent()}

def optimizer_mapper(optimizer):
    if type(optimizer) is str:
        if optimizer in optimizers_map:
            return optimizers_map[optimizer]
        else:
            raise ValueError(f"'{optimizer}' is not recognized")
    else:
        return optimizer