import pandas as pd

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from nn import *
from metrics import accuracy
from activations import Relu, Softmax
from tools import one_hot_encoder

def build_model():
    
    model = NeuralNetwork(loss=CrossEntropy(), n_iterations=800, learning_rate=0.13)
    model.add(Layer(4, 10, activation=Relu()))
    model.add(Layer(10, 3, activation=Softmax()))
    return model

def test_iris_accuracy(acc_threshold=.96):
    df = pd.read_csv('datasets/iris.csv', index_col=0)
    X, y = df.iloc[:, :-1], df.iloc[:,-1]
    y_ohe, label_map = one_hot_encoder(y)
    np.random.seed(42)
    model = build_model()
    model.fit(X, y_ohe)
    predictions = model.predict(X)
    pred_labels = [label_map[_] for _ in np.argmax(predictions, axis=1)]
    acc = accuracy(y, pred_labels)
    print(acc)
    assert acc > acc_threshold
    

if __name__ == '__main__':
    test_iris_accuracy()