import numpy as np

def accuracy(y_true, y_pred, threshold=.5, use_threshold=False):
        if use_threshold:
            ev = y_true == (y_pred > threshold)
        else:  # use argmax
            ev = np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1)
        return ev.sum() / ev.size

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def cv_score(model, scoring, X, y, cv=3, verbose=False):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    batch_size = len(indices) // cv
    cv_scores = []
    is_pd = 'iloc' in dir(X)
    for i in range(cv):
        idx_train = np.concatenate([indices[:batch_size * i], indices[batch_size * (i + 1):]])
        idx_test = indices[batch_size * i: batch_size * (i + 1)]
        
        X_train = X.iloc[idx_train, :] if is_pd else X[idx_train, :]
        X_test = X.iloc[idx_test, :] if is_pd else X[idx_test, :]
        
        y_train = y[idx_train]
        y_test = y[idx_test]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = scoring(y_test, y_pred)
        if verbose:
            print(f"[{i}]: {score}")
        cv_scores.append(score)
    return np.array(cv_scores)

metrics_available = [accuracy, mean_squared_error, rmse]

def metric_mapper(metric):
    if type(metric) is str:
        for m in metrics_available:
            if m.__name__ == metric:
                return m
    else:
        return metric