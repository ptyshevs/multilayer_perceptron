import numpy as np

class Loss:
    def __eq__(self, other):
        return repr(self) == repr(other)

class MSE(Loss):
    def forward(self, Y, Y_pred):
        return np.sum(np.power(Y - Y_pred, 2))
    
    def backward(self, Y, Y_pred):
        return -2 * (Y - Y_pred) / Y.shape[1]

    def __repr__(self):
        return 'mse'

class BinaryCrossEntropy(Loss):
    def forward(self, Y, Y_pred):
        Y_clipped = np.clip(Y_pred, 1e-12, None)
        return -(Y * np.log(Y_clipped) + (1 - Y) * np.log(1 - Y_clipped)).mean()
    
    def backward(self, Y, Y_pred):
#         Y_clipped = np.clip(Y_pred, 1e-12, None)
#         return - (np.divide(Y, Y_clipped) - np.divide(1 - Y, 1 - Y_clipped))
        return (Y_pred - Y) / (Y_pred * (1 - Y_pred))
    
    def __repr__(self):
        return "binary_crossentropy"

class CrossEntropy(Loss):
    def forward(self, Y, Y_pred):
        return -(Y * np.log(np.clip(Y_pred, 1e-12, None))).mean()
    
    def backward(self, Y, Y_pred):
        return Y_pred - Y
    
    def __repr__(self):
        return "crossentropy"
    
losses = {'mse':MSE,
          'binary_crossentropy': BinaryCrossEntropy,
          'crossentropy': CrossEntropy}

def loss_mapper(loss):
    if type(loss) is str:
        if loss in losses:
            return losses[loss]()
        else:
            raise ValueError(f"Loss is not recognized: {loss}")
    else:
        return loss