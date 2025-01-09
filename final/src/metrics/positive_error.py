import numpy as np


def positive_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the positive errors where the true values are greater than or equal to the predicted values.

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    np.ndarray: Array of positive errors where y_true >= y_pred.
    """
    above_bounds = y_true >= y_pred
    return y_true[above_bounds] - y_pred[above_bounds]