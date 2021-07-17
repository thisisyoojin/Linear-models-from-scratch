import numpy as np

def train_test_split(X, y, train_size=None, test_size=0.2):
    """
    Split arrays or matrices into random train and test subsets

    INPUT
    X: Numpy array of features - Matrix
    y: Numpy array of label - Vector
    """
    # Setup test size
    if train_size:
        test_size = 1 - train_size
    
    # Create an empty array to put test data
    X_test = np.empty((test_size, X.shape[1]))
    y_test = np.empty((test_size, 1))
    
    # Setup the iterating params for test data
    idx = 0
    test_data_count = len(X) * test_size
    
    # Iterate until it reaches to the total number of test data
    while idx < test_data_count:
        # Generates random index 
        generated_num = np.random.randint(0, len(X))
        # Put the data in random index to test data array
        X_test[idx] = X[generated_num]
        y_test[idx] = y[generated_num]
        # Remove the data in random index in train data array
        X = np.delete(X, generated_num, axis=0)
        y = np.delete(y, generated_num, axis=0)
        # Increment index
        idx += 1

    return X, X_test, y, y_test