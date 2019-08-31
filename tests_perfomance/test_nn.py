from nn import *

def test_layer():
    L = Layer(2, 2)
    L.W = np.array([[.35, .7],
                    [.76, .63]])
    L.b = np.array([[2],
                    [-1]])
    X = np.array([[1],
                  [2]])
    
    r = L.forward_propagate(X)

    assert (r == np.array([[3.75], [1.02]])).all()

    rb = L.backward_propagate(1)
    