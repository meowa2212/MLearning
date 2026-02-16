import numpy as np

class Loss():
    class MSE():
        '''regression (standard)'''
        def calc(self, y_pred: np.ndarray, y_true: np.ndarray):
            return np.mean((y_pred - y_true)**2)
        def deriv(self, y_pred: np.ndarray, y_true: np.ndarray):
            return 2 * (y_pred - y_true) / y_true.size

    class MAE():
        '''regression (robust with outliers)'''
        def calc(self, y_pred: np.ndarray, y_true: np.ndarray):
            return np.mean(np.abs(y_pred - y_true))
        def deriv(self, y_pred: np.ndarray, y_true: np.ndarray):
            return np.sign(y_pred - y_true) / y_true.size

    class BCE():
        '''binary classification (use with Sigmoid)'''
        def calc(self, y_pred: np.ndarray, y_true: np.ndarray):
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        def deriv(self, y_pred: np.ndarray, y_true: np.ndarray):
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            return (y_pred - y_true) / (y_pred * (1 - y_pred))

    class CCE():
        '''multiclass (use with softmax)'''
        def calc(self, y_pred: np.ndarray, y_true: np.ndarray):
            return -np.sum(y_true * np.log(y_pred + 1e-15)) / y_true.shape[1]
        
        def deriv(self, y_pred: np.ndarray, y_true: np.ndarray):
            '''
            This is the combined gradient of Softmax + CCE.
            When using this, the final layer's activation derivative 
            should be treated as 1 (Identity).
            '''
            return (y_pred - y_true) / y_true.shape[1]