import numpy as np

class LinearModel:

    def __init__(self, learning_rate=0.01, batch_size=16):
        """
        Initialise linear regression model
        """
        self.w = None
        self.b = None
        self.learning_rate = learning_rate
        self.batch_size = batch_size


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


    def predict(self, X):
        """
        Predict the values
        """
        if self.w is None:
            raise Exception("You should fit a model first.")
        return np.matmul(X, self.w) + self.b        


    def calculate_gradient(self, X, y, y_pred):
        """
        Calculate gradient for the current parameters
        """
        # predict the value with a model

        # calculate the gradient for weights(coefficient)
        grad_individuals = []
        for idx in range(len(X)):
            grad = 2 * (y_pred[idx] - y[idx]) * X[idx]
            grad_individuals.append(grad)
        grad_w = np.mean(grad_individuals, axis=0)
        
        # calculate the gradient for bias
        grad_b = 2 * np.mean(y_pred - y, axis=0)

        return grad_w, grad_b
