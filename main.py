from sklearn.datasets import load_boston
from model_selection import train_test_split, GridSearchCV
from linear_model import LinearRegression
import numpy as np

def preprocess(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    return X_train, X_val, X_test, y_train, y_val, y_test


def regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y)
    print(X_train.shape, X_val.shape, X_test.shape)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    print(regressor.score(X_val, y_val))


def grid_search():
    params = {
        "learning_rate": [0.01, 0.05, 0.1],
        "batch_size": [8, 16, 32]
    }
    lr = GridSearchCV(LinearRegression, params)
    X, y = load_boston(return_X_y=True)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess(X, y)
    results = lr.fit(X_train, y_train)
    for result in results:
        print(f"{result[0]} - Training loss: {np.mean(result[1])}")
    best_result = sorted(results, key=lambda x:x[1])[0]
    print(f"Best hyperparameter is: {best_result[0]}")


if __name__ == "__main__":
    #regression()
    grid_search()
    

    
