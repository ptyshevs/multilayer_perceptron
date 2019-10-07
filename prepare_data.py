# This script will split dataset into train and test parts

import pandas as pd
from mlp.tools import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('resources/data.csv', header=None)
    train, test = train_test_split(df.values, random_state=42)
    col_names = ['id', 'class'] + ['feature_' + str(_) for _ in range(len(df.columns) - 2)]
    df_train = pd.DataFrame(train, columns=col_names)
    df_test = pd.DataFrame(test, columns=col_names)
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    