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
        return f"weight:{self.w}\nbias:{self.b}"


    def fit(self, X_train, y_train, epochs=10, learning_rate=0.001):
        """
        Fit the model according to the given training data
        """
        # Initialise the parameters for model
        n_features = X_train.shape[1]
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()
        
        for epoch in range(epochs):
            y_pred = self.predict(X_train)
            grad_w, grad_b = self.calculate_gradient(X_train, y_train, y_pred)
            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b
            loss = self.calculate_loss(y_train, y_pred)
            print(f"Loss of epoch {epoch+1}: {loss}")
        


    def predict(self, X):
        """
        Predict the values
        """
        if self.w is None:
            raise Exception("You should fit a model first.")
        return np.matmul(X, self.w) + self.b


    def calculate_loss(self, y, y_pred):
        """
        MSE(Mean Squared Error)
        """
        return np.mean((y_pred - y)**2)


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
        grad_b = 2 * np.mean(y_pred - y)

        return grad_w, grad_b


    def score(self, X, y_true):
        """
        Return the coefficient of determination R squared of the prediction
        """
        y_pred = self.predict(X)
        # u is the residual sum of squres
        u = ((y_true - y_pred) ** 2).sum()
        # v is the total sum of squares
        v = ((y_true - y_true.mean()) ** 2).sum()
        return round(1 - u/v, 5)