import pandas as pd
import numpy as np

import argparse

from mlp.nn import *
from mlp.viz import plot_history
from mlp.callbacks import EarlyStopping, ModelCheckpoint
from mlp.metrics import accuracy, precision, recall, cv_score
from mlp.tools import train_test_split, load, save, one_hot_encoder, binary_encoder, normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='resources/data.csv', help='dataset to train model for')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose training process')
    parser.add_argument('--file', '-f', default='weights.pcl', help='filename for loading model')
    parser.add_argument('--seed', '-s', default=42, help='random seed')
    parser.add_argument('--cross_validation', '-c', default=False, action='store_true', help='Evaluate using cross-validation')
    parser.add_argument('--softmax', default=False, action='store_true', help='softmax (Required by subject)')

    args = parser.parse_args()

    if type(args.seed) is str:
        args.seed = int(args.seed)

    df = pd.read_csv(args.dataset, index_col=0)
    

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    if args.softmax:
        y, ohe_map = one_hot_encoder(y)
    else:
        y, ohe_map = binary_encoder(y)

    model = load(args.file)
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=.33, random_state=args.seed)
    X_train, X_mean, X_std = normalize(X_train)
    X_test, _, _ = normalize(X_test, X_mean, X_std)
    print("Accuracy on test dataset:", accuracy(y_test, model.predict(X_test)))
    if args.cross_validation:
        X, _, _ = normalize(X.values)
        cv = cv_score(model, accuracy, X, y, cv=5, verbose=True, n_epochs=model.n_epochs, batch_size=model.batch_size)
        print(f"Mean CV accuracy: {np.mean(cv)} | Std CV accuracy: {np.std(cv)}")