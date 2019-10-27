import pandas as pd
import argparse

from mlp.nn import *
from mlp.viz import plot_history
from mlp.callbacks import EarlyStopping, ModelCheckpoint
from mlp.metrics import accuracy, precision, recall
from mlp.tools import train_test_split, load, save, one_hot_encoder, binary_encoder, normalize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', default='resources/data.csv', help='dataset to train model for')
    parser.add_argument('-l', '--learning_rate', default=.05, type=float, help='Specify learning rate')
    parser.add_argument('-v', '--verbose', default=False, action='store_true', help='Verbose training process')
    parser.add_argument('--file', '-f', default='weights.pcl', help='filename for saving model')
    parser.add_argument('--epochs', '-e', default=200, help='# of epochs')
    parser.add_argument('--batch-size', '-b', default=32, help='Batch size')
    parser.add_argument('--seed', '-s', default=42, help='Random seed')
    parser.add_argument('--softmax', default=False, help='use softmax (required by subject)', action='store_true')
    
    args = parser.parse_args()

    if type(args.learning_rate) is str:
        args.learning_rate = float(args.learning_rate)
    if type(args.epochs) is str:
        args.epochs = int(args.epochs)
    if type(args.batch_size) is str:
        args.batch_size = int(args.batch_size)
    if type(args.seed) is str:
        args.seed = int(args.seed)
    
    df = pd.read_csv(args.dataset, index_col=0)
    
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    
    if args.softmax:
        y, ohe_map = one_hot_encoder(y)
        
        model = NeuralNetwork(loss='crossentropy', optimizer=Adam(lr=args.learning_rate), verbose=args.verbose,
                            verbose_step=20, random_state=args.seed)
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=.33, random_state=args.seed)
        
        X_train, X_mean, X_std = normalize(X_train)
        X_test, _, _ = normalize(X_test, X_mean, X_std)
        model.compile(X_train.shape[1])

        h = model.fit(X_train, y_train, X_test, y_test, n_epochs=args.epochs, batch_size=args.batch_size, metrics=['accuracy'])
        plot_history(h)
        print("Accuracy on test dataset:", accuracy(y_test, model.predict(X_test)))
    else:
        y, ohe_map = binary_encoder(y)
        
        model = NeuralNetwork(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), verbose=args.verbose,
                            verbose_step=20, random_state=args.seed)
        model.add(Dense(32, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=.33, random_state=args.seed)
        
        X_train, X_mean, X_std = normalize(X_train)
        X_test, _, _ = normalize(X_test, X_mean, X_std)
        model.compile(X_train.shape[1])

        h = model.fit(X_train, y_train, X_test, y_test, n_epochs=args.epochs, batch_size=args.batch_size, metrics=['accuracy', 'precision'])
        plot_history(h)
        print("Accuracy on test dataset:", accuracy(y_test, model.predict(X_test)))
    
    save(model, args.file)