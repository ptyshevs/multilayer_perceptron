import numpy as np
from tools import train_test_split

def test_train_test_split():
    X_train, X_test = train_test_split(np.array([0, 1, 2]), random_state=42)
    assert X_train == np.array([0, 1])
    assert X_test == np.array([2])