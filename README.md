## Multilayer Perceptron

Pure-`numpy` version of `Keras` for educational purposes.

## Usage

```python
from nn import NeuralNetwork, Dense
from tools import one_hot_encoder
from metrics import cv_score, accuracy
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

y_enc, _ = one_hot_encoder(y)

model = NeuralNetwork(loss='crossentropy')
model.add(Dense(3, activation='softmax'))
model.fit(X, y_enc, n_epochs=50)
```