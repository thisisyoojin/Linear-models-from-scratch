from model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from linear_model import Batchifier, LinearModel

class LinearRegression(LinearModel):
    
    def __init__(self, learning_rate=0.01, batch_size=16):
        """
        Initialise linear regression model
        """
        super().__init__(learning_rate=learning_rate, batch_size=batch_size)


    def fit(self, X_train, y_train, normalise=False, epochs=30, learning_rate=None, batch_size=None, draw=False, debug=True):
        """
        Fit the model according to the given training data
        """
        # Creates a validation dataset for training
        if normalise:
            X_train, X_val, y_train, y_val = self.normalise_train_data(X_train, y_train)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

        # Set up the hyperparameters for a model
        if learning_rate is None:
            learning_rate = self.learning_rate

        if batch_size is None:
            batch_size = self.batch_size
        
        # Initialise the parameters for model
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()
        
        
        # List for losses and early stopping loss
        train_losses = []
        val_losses = []
        
        # Create an instance for batchifier
        batchifier = Batchifier(batch_size=batch_size)
        
        # epochs: the number of running the whole dataset
        for epoch in range(epochs):
            
            losses_per_epoch = []
            
            # Creates a shuffled batch 
            batchifier.batch(X_train, y_train)

            # Train with batch
            for X_batch, y_batch in batchifier:
                y_pred = self.predict(X_batch)
                # Calculates the gradient and update params
                grad_w, grad_b = self.calculate_gradient(X_batch, y_batch, y_pred)
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
                # Calculates the loss
                loss_per_batch = self.calculate_loss(y_batch, y_pred)
                losses_per_epoch.append(loss_per_batch)
            
            train_losses.append(np.mean(losses_per_epoch))
            
            if debug:
                print(f"Loss of epoch {epoch+1}: {np.mean(losses_per_epoch)}")

            # Validation loss
            y_val_pred = self.predict(X_val)
            val_loss = self.calculate_loss(y_val, y_val_pred)
            val_losses.append(val_loss)


        if draw:
            plt.plot(train_losses)
            plt.plot(val_losses)
            plt.show()

        return train_losses, val_losses



    def calculate_gradient(self, X, y, y_pred):
        """
        Calculate gradient for the current parameters
        """

        # calculate the gradient for weights(coefficient)
        # grad_individuals = []
        # for idx in range(len(X)):
        #     grad = 2 * (y_pred[idx] - y[idx]) * X[idx] 
        #     grad_individuals.append(grad)
        # grad_w = np.mean(grad_individuals, axis=0)        

        # calculate the gradient for bias
        grad_w = 2 * (y_pred - y) @ X / len(X)
        grad_b = 2 * np.mean(y_pred - y, axis=0)

        return grad_w, grad_b


    def calculate_loss(self, y, y_pred):
        """
        MSE(Mean Squared Error)
        """
        return np.mean((y_pred - y)**2)


    def predict(self, X):
        """
        Predict the values
        """
        if self.w is None:
            raise Exception("You should fit a model first.")
        return np.matmul(X, self.w) + self.b


    def score(self, X_test, y_test, noramlise=True):
        """
        Return the coefficient of determination R squared of the prediction
        """
        if noramlise:
            X_test = (X_test - self.X_train_mean) / self.X_train_std
        
        y_pred = self.predict(X_test)
        # u is the residual sum of squres
        u = ((y_test - y_pred)**2).sum()
        # v is the total sum of squares
        v = ((y_test - y_test.mean()) ** 2).sum()
        return round(1 - u/v, 5)