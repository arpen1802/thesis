import numpy as np
from src.metrics.positive_error import positive_error


def absolute_percentage_positive_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """
    Calculate the relative positive error where the true values are greater or equal to the predicted values.

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    np.ndarray: Array of relative positive errors.
    """
    threshold = 1e-2
    above_bounds = y_true >= y_pred
    y_true[above_bounds] = np.maximum(y_true[above_bounds], threshold)
    return (positive_error(y_true, y_pred) / y_true[above_bounds]) * 100
