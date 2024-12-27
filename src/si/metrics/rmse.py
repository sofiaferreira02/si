import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
    - y_true (numpy.ndarray or list): Real values of y.
    - y_pred (numpy.ndarray or list): Predicted values of y.

    Returns:
    - float: RMSE, the square root of the mean of squared differences between y_true and y_pred.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))

