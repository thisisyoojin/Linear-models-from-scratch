import numpy as np
from model_selection import train_test_split

class LinearModel:

    def __init__(self, learning_rate=None, batch_size=None):
        """
        Initialise linear regression model
        """
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.X_train_mean = None
        self.X_train_std = None


    @property
    def coef_(self):
        """
        Get weight(coefficient) for this estimator
        """
        return self.w
    
    @property
    def intercept_(self):
        """
        Get bias(intercept) for this estimator
        """
        return self.b

    
    def normalise_train_data(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

        self.X_train_mean = np.mean(X_train, axis=0)
        self.X_train_std = np.std(X_train, axis=0)
        
        X_train = (X_train - self.X_train_mean) / self.X_train_std
        X_val = (X_val - self.X_train_mean) / self.X_train_std
        
        return X_train, X_val, y_train, y_val


    def normalise_test_data(self, X_test, y_test):
        
        if self.X_train_mean is None:
            raise Exception("You need to fit train dataset with normalise option")
        
        X_test = (X_test - self.X_train_mean) / self.X_train_std
        return X_test, y_test


