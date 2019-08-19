## Multilayer Perceptron

Pure-`numpy` version of `Keras` for educational purposes.

## Usage

```python
from nn import NeuralNetwork
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = NeuralNetwork(loss='crossentropy')
model.add(Dense(4, 3, activation='softmax'))

model.fit(X, y)
```