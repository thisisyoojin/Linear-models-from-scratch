import numpy as np

class LinearRegression:
    
    def __init__(self):
        """
        Initialise linear regression model
        """
        self.w = None
        self.b = None


    def get_params(self):
        """
        Get parameters for this estimator
        """
        pass


    def fit(self, X, y):
        """
        Fit the model according to the given training data
        """
        pass

    def predict(self, X):
        """
        Predict the values
        """
        if not self.w:
            raise Exception("You should fit a model first.")
        return np.matmul(X, self.w) + self.b


    def calculate_loss(self):
        """
        Optimisor for the linear regression
        """
        pass


    def calculate_gradient(self):
        """
        Calculate gradient for the current parameters
        """
        pass


    def score(self):
        """
        Return the coefficient of determination R squared of the prediction
        """
        pass