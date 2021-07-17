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
        # Initialise the parameters for model
        n_features = X.shape[1]
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()
        
        pass


    def predict(self, X):
        """
        Predict the values
        """
        if not self.w:
            raise Exception("You should fit a model first.")
        return np.matmul(X, self.w) + self.b


    def calculate_loss(self, X, y):
        """
        MSE(Mean Squared Error)
        """
        pass


    def calculate_gradient(self, X, y):
        """
        Calculate gradient for the current parameters
        """
        # predict the value with a model
        y_pred = self.predict(X)

        # calculate the gradient for weights(coefficient)
        grad_individuals = []
        for idx in range(len(X)):
            grad = 2 * (y_pred[idx] - y[idx]) * X[idx]
            grad_individuals.append(grad)
        grad_w = np.mean(grad_individuals, axis=0)
        
        # calculate the gradient for bias
        grad_b = 2 * np.mean(y_pred - y)

        return grad_w, grad_b


    def score(self):
        """
        Return the coefficient of determination R squared of the prediction
        """
        pass