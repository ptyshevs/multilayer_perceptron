import pandas as pd

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from nn import *
from metrics import rmse
from activations import Tanh
from tools import one_hot_encoder, normalize

def build_model():
    m = NeuralNetwork(n_iterations=900)
    m.add(Layer(11, 11, activation=Tanh()))
    m.add(Layer(11, 1))
    return m

def test_wine_rmse(rmse_threshold=660):
    df = pd.read_csv('datasets/wine.csv')
    X, y = df.iloc[:, :-1].values, df.iloc[:,-1].values[:, np.newaxis]
    X = normalize(X)
    np.random.seed(42)
    model = build_model()
    model.fit(X, y)
    predictions = model.predict(X)
    loss = rmse(y, predictions)
    assert loss < rmse_threshold
    

if __name__ == '__main__':
    test_wine_rmse()