from sklearn.datasets import load_boston
from model_selection import train_test_split
from linear_model import LinearRegression

def regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)
    regressor = LinearRegression()


if __name__ == "__main__":
    regression()
    
