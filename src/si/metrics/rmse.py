import numpy as np
from cmath import sqrt

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    - y_true (numpy.ndarray or list): Real values of y.
    - y_pred (numpy.ndarray or list): Predicted values of y.

    Returns:
    - float: RMSE, the square root of the mean of squared differences between y_true and y_pred.
    """
    # Ensure that the inputs are NumPy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse_value = sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

    return rmse_value