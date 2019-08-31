import numpy as np
import pickle as pcl

def binary_encoder(y):
    """
    For binary classification, use single binary feature. Otherwise, send to OHE.
    """
    y_unique = np.unique(y)
    if len(y_unique) != 2:
        print(f"There are {len(y_unique)} unique labels, using OHE")
        return one_hot_encoder(y)
    M = np.zeros((len(y), 1))
    M[y == y_unique[-1]] = 1
    label_map = {i:label for i, label in enumerate(y_unique)}
    return M, label_map

def one_hot_encoder(y):
    """
    Take vector of labels, return one-hot-encoded version along with label map
    """
    y_unique = np.unique(y)
    M = np.zeros((len(y), len(y_unique)))
    for i, v in enumerate(y_unique):
        M[y == v,i] = 1
    label_map = {i:label for i, label in enumerate(y_unique)}
    return M, label_map

def one_hot_decoder(ohe, label_map):
    """
    Convert one-hot encoded variables back into their labels.
    
    Find where ones are, then take column indices and map back to labels.
    
    @param ohe: 2D array, where each row contains zeros everywhere except for it's label, which is one.
    @param label_map: index -> label mapping
    """
    return np.array([label_map[_] for _ in np.where(ohe == 1)[1]])

def argmax_to_label(prediction, label_map):
    return np.array([label_map[_] for _ in np.argmax(predictions, axis=1)])[:, np.newaxis]

def normalize(X):
    """
    Normalize, i.e. standardize dataset.
    
    Each column is centered around zero and has unit variance
    """
    return (X - X.mean(axis=0)) / X.std(axis=0)

def train_test_split(*args, test_size=.25, random_state=None):
    """
    Arguments are shuffled (consistently, using indices) and then split into train-test parts.
    """
    if random_state is not None:
        np.random.seed(random_state)
    n = len(args[0])
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    train_indices_size = int((1 - test_size) * n)
    train_indices = indices[:train_indices_size]
    test_indices = indices[train_indices_size:]
    
    res = []
    for arg in args:
        res.append(arg[train_indices, :])
        res.append(arg[test_indices, :])
    return res

def save(model, path):
    with open(path, 'wb') as fp:
        fp.write(pcl.dumps(model))
    
def load(path):
    with open(path, 'rb') as fp:
        return pcl.loads(fp.read())