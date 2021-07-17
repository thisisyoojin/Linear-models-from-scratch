from sklearn.datasets import load_boston
from model_selection import train_test_split
from linear_model import LinearRegression
import numpy as np

def regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_val = (X_val - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    regressor = LinearRegression()
    regressor.fit(X_train, y_train, draw=True)
    print(regressor.score(X_train, y_train))


if __name__ == "__main__":
    regression()
    
