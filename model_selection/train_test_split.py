import numpy as np

def train_test_split(X, y, train_size=None, test_size=0.2):
    """
    Split arrays or matrices into random train and test subsets

    INPUT
    X: Numpy array of features - Matrix
    y: Numpy array of label - Vector
    """

    if train_size:
        test_size = 1 - train_size

    idx = 0
    test_data_count = len(X) * test_size
    X_test = np.empty(X.shape[1])
    y_test = np.empty(1)

    while idx < test_data_count:
        generated_num = np.random.randint(0, len(X))
        X_test = np.vstack((X_test, X[generated_num]))
        y_test = np.vstack((y_test, y[generated_num]))
        np.delete(X, generated_num, axis=0)
        np.delete(y, generated_num, axis=0)
        idx += 1

    return X, X_test, y, y_test