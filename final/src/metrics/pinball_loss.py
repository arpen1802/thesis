import numpy as np
from sklearn.metrics import mean_pinball_loss


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Calculate the mean pinball loss.

    Parameters:
    y_true : array-like of shape (n_samples,)
        The true target values.
    quantile_preds : array-like of shape (n_samples,)
        The predicted quantile values.
    alpha : float
        The quantile level.

    Returns:
    float
        The mean pinball loss.
    """
    return mean_pinball_loss(y_true=y_true, y_pred=y_pred, alpha=alpha)
