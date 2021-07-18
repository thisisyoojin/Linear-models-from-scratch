from linear_model.LinearModel import LinearModel
import numpy as np
import matplotlib.pyplot as plt
from .Batchifier import Batchifier

class LinearRegression(LinearModel):
    
    def __init__(self, learning_rate=0.01, batch_size=16):
        """
        Initialise linear regression model
        """
        super().__init__(learning_rate=learning_rate, batch_size=batch_size)


    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=50, learning_rate=None, batch_size=None, draw=False):
        """
        Fit the model according to the given training data
        """
        
        # Initialise the parameters for model
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()
        
        
        if learning_rate is None:
            learning_rate = self.learning_rate

        if batch_size is None:
            batch_size = self.batch_size
        

        # List for losses and early stopping loss
        train_losses = []
        val_losses = []
        last_5_losses = [0]*5
        
        # Create an instance for batchifier
        batchifier = Batchifier()
        
        # epochs: the number of running the whole dataset
        for epoch in range(epochs):
            loss_per_epoch = []
            
            # Creates a shuffled batch 
            batchifier.batch(X_train, y_train)

            for X_batch, y_batch in batchifier:
                # Predict the value
                y_pred = self.predict(X_batch)
                grad_w, grad_b = self.calculate_gradient(X_batch, y_batch, y_pred)
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
                loss_per_batch = self.calculate_loss(y_batch, y_pred)
                loss_per_epoch.append(loss_per_batch)

            print(f"Loss of epoch {epoch+1}: {np.mean(loss_per_epoch)}")
            train_losses.append(np.mean(loss_per_epoch))

            if X_val is not None:
                y_pred = self.predict(X_val)
                val_loss = self.calculate_loss(y_val, y_pred)
                val_losses.append(val_loss)
                last_5_losses[epoch%5] = val_loss

            else:
                last_5_losses[epoch%5] = np.mean(loss_per_epoch)

            # Early stopping
            if np.std(last_5_losses) <= 1:
                print("No obvious improvment is observed.")
                break

           
        if draw:
            plt.plot(train_losses)
            if X_val is not None:
                plt.plot(val_losses)
            plt.show()

        return train_losses, val_losses



    def calculate_loss(self, y, y_pred):
        """
        MSE(Mean Squared Error)
        """
        return np.mean((y_pred - y)**2)



    def score(self, X, y_true):
        """
        Return the coefficient of determination R squared of the prediction
        """
        y_pred = self.predict(X)
        # u is the residual sum of squres
        u = ((y_true - y_pred)**2).sum()
        # v is the total sum of squares
        v = ((y_true - y_true.mean()) ** 2).sum()
        print("u:",u, ",v:", v, ", u/v:", u/v)
        return round(1 - u/v, 5)