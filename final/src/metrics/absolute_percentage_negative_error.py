import numpy as np
from src.metrics.negative_error import negative_error


def absolute_percentage_negative_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate the relative negative error where the true values are less than the predicted values.

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    np.ndarray: Array of relative negative errors.
    """
    threshold = 1e-2
    above_bounds = y_true >= y_pred
    # y_true[~above_bounds] = np.maximum(y_true[~above_bounds], threshold)
    return (negative_error(y_true, y_pred) / np.maximum(y_true[~above_bounds], threshold)) * 100
