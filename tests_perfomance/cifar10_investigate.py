import numpy as np
import sys
from nn import *
from tools import train_test_split, one_hot_encoder
from viz import plot_history

data = np.load('datasets/cifar10.npz')
X = data['arr_0']
y = data['arr_1']
X = X.astype(np.float32) / 255. - .5
y, _ = one_hot_encoder(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

np.random.seed(42)
m = NeuralNetwork('crossentropy', optimizer=Adam(lr=0.05), verbose=True, verbose_step=1, debug=True,
                   random_state=42)
m.add(Flatten())
m.add(Dense(256, 'elu', weights_regularizer=L2Regularizer()))
m.add(BatchNorm())
m.add(Dense(10, 'softmax'))
# Just to initialize parameters
m.fit(X_train, y_train, X_test, y_test, n_epochs=0, batch_size=256, metrics=['accuracy'])
m.summary()
# Actual training
h = m.fit(X_train, y_train, X_test, y_test, n_epochs=6, batch_size=256, metrics=['accuracy'])
