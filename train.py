import pandas as pd
import argparse

from mlp.nn import *
from mlp.viz import plot_history
from mlp.callbacks import EarlyStopping, ModelCheckpoint
from mlp.metrics import accuracy
from mlp.tools import train_test_split, load, save, binary_encoder, normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='resources/data.csv', help='dataset to train model for')
    parser.add_argument('-l', '--learning_rate', default=.05, type=float, help='Specify learning rate')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose training process')
    parser.add_argument('--save-path', '-s', default='weights.pcl', help='filename for saving model')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.dataset, index_col=0)
    
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    
    y, ohe_map = binary_encoder(y)
    
    model = NeuralNetwork(loss='binary_crossentropy', optimizer=Adam(lr=0.03), verbose=args.verbose, verbose_step=20)
    model.add(Dense(30, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=.33, random_state=42)
    
    X_train, X_mean, X_std = normalize(X_train)
    X_test, _, _ = normalize(X_test, X_mean, X_std)
    model.compile(X_train.shape[1])

    print(X_train.shape, y_train.shape, model.layers[0].W.shape)
    h = model.fit(X_train, y_train, X_test, y_test, n_epochs=200, batch_size=32, metrics=['accuracy'])
    plot_history(h)
    print("Accuracy on test dataset:", accuracy(y_test, model.predict(X_test)))
    
    save(model, args.save_path)