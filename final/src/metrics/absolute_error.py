import numpy as np


def absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute error between true and predicted values.

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    np.ndarray: Array of absolute errors.
    """
    return np.abs(y_true - y_pred)
