import numpy as np
import itertools

class GridSearchCV:
    """
    Grid search class to find a best set of hyperparameters
    """
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid

    def fit(self, X, y, param_grid=None):
        """
        Run fit with all sets of parameters
        """
        if param_grid is None:
            param_grid = self.param_grid
        
        keys = list(self.param_grid.keys())
        values = self.param_grid.values()
        res = list(itertools.product(*values))

        result = []
        
        for r in res:
            param = {keys[idx]:r[idx] for idx in range(len(r))}
            train_loss, val_loss = self.model().fit(X, y, **param)
            result.append((param, train_loss, val_loss))

        return result
            
        
        


            
        
            
        
        
                        
            
        
        

            
        
        
        #estimator.fit(X, y, learning_rate=)
        #print(k)

        