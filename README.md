## Multilayer Perceptron

Pure-`numpy` version of `Keras` for educational purposes.

## Features

* Keras-like model fitting procedure in `nn.py`
* Activation functions (ReLU, ELU, Softmax) in `activations.py`
* Early Stopping and ModelCheckpoint callbacks in `callbacks.py`
* Multiple layers available (Dense, BatchNorm, Dropout) in `layers.py`
* Different Optimizers (Momentum SGD, RMSProp, Adam) and learning-rate schedulers in `optim.py`

## Usage

```python
import mlp
from mlp.nn import NeuralNetwork, Dense
from mlp.tools import one_hot_encoder
from mlp.metrics import cv_score, accuracy
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

y_enc, _ = one_hot_encoder(y)

model = NeuralNetwork(loss='crossentropy')
model.add(Dense(3, activation='softmax'))
model.fit(X, y_enc, n_epochs=50)
```