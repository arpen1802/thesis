import numpy as np
from src.metrics.absolute_error import absolute_error


def absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the relative error between the true values and the predicted values.

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    np.ndarray: Array of relative errors.
    """
    threshold = 1e-2
    y_true_adj = np.maximum(y_true, threshold)
    return (absolute_error(y_true, y_pred) / y_true_adj) * 100
