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

