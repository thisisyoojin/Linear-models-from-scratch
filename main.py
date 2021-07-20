from sklearn.datasets import load_boston, load_breast_cancer
from model_selection import train_test_split, GridSearchCV
from linear_model import LinearRegression, LogisticRegression
import numpy as np

np.set_printoptions(precision=10)


def regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train, normalise=True, draw=True)
    print(regressor.score(X_test, y_test))


def grid_search():

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    params = {
        "learning_rate": [0.01, 0.05, 0.1],
        "batch_size": [8, 16, 32]
    }
    lr = GridSearchCV(LinearRegression, params)
    results = lr.fit(X_train, y_train)
    for result in results:
        print(f"{result[0]} - Training loss: {np.mean(result[1])}")
    best_result = sorted(results, key=lambda x:x[1])[0]
    print(f"Best hyperparameter is: {best_result[0]}")


def classification():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)


if __name__ == "__main__":
    #regression()
    #grid_search()
    classification()

    
