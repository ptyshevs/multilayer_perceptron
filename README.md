## Multilayer Perceptron

Pure-`numpy` version of `Keras` for educational purposes.

## Usage

```python
from nn import NeuralNetwork, Layer, CrossEntropy
from activations import Relu, Softmax
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = NeuralNetwork(loss=CrossEntropy())
model.add(Layer(4, 3, activation=Softmax()))

model.fit(X, y)
```