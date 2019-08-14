import numpy as np

class MSE:
    def forward(self, Y, Y_pred):
        return np.sum(np.power(Y - Y_pred, 2))
    
    def backward(self, Y, Y_pred):
        return -2 * (Y - Y_pred) / Y.shape[1]

    def __repr__(self):
        return 'MSE'

class Binary:
    def forward(self, Y, Y_pred):
        Y_clipped = np.clip(Y_pred, 1e-12, None)
        return -(Y * np.log(Y_clipped) + (1 - Y) * np.log(Y_clipped)).mean()
    
    def backward(self, Y, Y_pred):
        Y_clipped = np.clip(Y_pred, 1e-12, None)
        return - (np.divide(Y, Y_clipped) - np.divide(1 - Y, 1 - Y_clipped))
    
    def __repr__(self):
        return "Binary cross-entropy"

class CrossEntropy:
    def forward(self, Y, Y_pred):
        return -(Y * np.log(np.clip(Y_pred, 1e-12, None))).mean()
    
    def backward(self, Y, Y_pred):
        return Y_pred - Y
    
    def __repr__(self):
        return "Multinomial cross-entropy"