import numpy as np


def coverage_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the coverage accuracy, which is the proportion of true values
    that are below the predicted values.

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    np.ndarray: The mean proportion of true values that are below
                the predicted values.
    """
    above_bounds = y_true >= y_pred
    return np.mean(~above_bounds) * 100
