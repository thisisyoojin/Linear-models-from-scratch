import math
import numpy as np
import matplotlib.pyplot as plt
from linear_model import LinearModel, Batchifier
from model_selection import train_test_split



class LogisticRegression(LinearModel):
    
    def __init__(self, learning_rate=0.01, batch_size=64):
        """
        Initialise Logistic regression model
        """
        super().__init__(learning_rate=learning_rate, batch_size=batch_size)
        

    def fit(self, X_train, y_train, normalise=True, learning_rate=None, batch_size=None, epochs=20, debug=True, draw=True):
        """
        Fit the model according to the given training data
        """

        # Creates a validation dataset for training
        if normalise:
            X_train, X_val, y_train, y_val = self.normalise_train_data(X_train, y_train)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train)

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
        
        # Create an instance for batchifier
        batchifier = Batchifier(batch_size=batch_size)
        
        # epochs: the number of running the whole dataset
        for epoch in range(epochs):
            loss_per_epoch = []
            
            # Creates a shuffled batch 
            batchifier.batch(X_train, y_train)

            for X_batch, y_batch in batchifier:
                # Predict the value which is needed when calculating gradient and loss
                y_pred = self.predict_proba(X_batch)
                grad_w, grad_b = self.calculate_gradient(X_batch, y_batch, y_pred)
                self.w -= learning_rate * grad_w
                self.b -= learning_rate * grad_b
                loss_per_batch = self.calculate_loss(y_batch, y_pred)
                loss_per_epoch.append(loss_per_batch)
                
                # debug printing to check prediction on the last epoch
                if epoch == epochs-1:
                    for y_hat, y_true in zip(y_pred, y_batch):
                        print(y_hat, ":", y_true)

            if debug:
                print(f"Loss of epoch {epoch+1}: {np.mean(loss_per_epoch, axis=0)}")
            
            train_losses.append(np.mean(loss_per_epoch))

            
            y_pred = self.predict_proba(X_val)
            val_loss = self.calculate_loss(y_val, y_pred)
            val_losses.append(val_loss)

            
        if draw:
            plt.plot(train_losses)
            if X_val is not None:
                plt.plot(val_losses)
            plt.show()

        return train_losses, val_losses


    def calculate_loss(self, y, y_pred):
        """
        Criteria: Loss function for the linear regression
        """
        losses = - ( y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return np.mean(losses)


    def calculate_gradient(self, X, y, y_pred):
        """
        Optimisor: Calculate the gradients in the perspective of params
        """
        # calculate the gradient for weights(coefficient)
        dL_dy_pred = -(y/y_pred - (1-y)/(1-y_pred))
        dy_pred_dz = y_pred * (1-y_pred)

        # grad_individuals = []
        # for idx in range(len(X)):
        #     grad = dL_dy_pred[idx] * dy_pred_dz[idx] * X[idx]
        #     grad_individuals.append(grad)

        # grad_w = np.mean(grad_individuals, axis=0)
        
        # calculate the gradient for bias
        grad_w = (dL_dy_pred * dy_pred_dz) @ X / len(X)
        grad_b = np.dot(dL_dy_pred, dy_pred_dz)

        return grad_w, grad_b



    def predict_logits(self, X):
        """
        Calculate logits(unnormalised probability)
        """
        return np.matmul(X, self.w) + self.b


    def predict_proba(self, X):
        """
        Calculate probability estimates from logits
        """
        logits = self.predict_logits(X)
        probas = []
        for logit in logits:
            proba = 1 / (1 + math.exp(-logit))
            probas.append(proba)
        return np.array(probas)


    def predict(self, X):
        proba = self.predict_proba(X)
        decision_function = lambda p: 1 if p >= 0.5 else 0
        prediction = decision_function(proba)
        return prediction



    def score(self):
        """
        Return the coefficient of determination R squared of the prediction
        """
        pass