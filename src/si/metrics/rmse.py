import numpy as np


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

    rmse_value = np.sqrt(np.mean((y_true - y_pred) ** 2))

    return rmse_value

