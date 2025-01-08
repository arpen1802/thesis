import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.metrics.absolute_percentage_error import absolute_percentage_error
from src.metrics.absolute_percentage_negative_error import absolute_percentage_negative_error
from src.metrics.absolute_percentage_positive_error import absolute_percentage_positive_error
from src.metrics.negative_error import negative_error
from src.metrics.positive_error import positive_error
from src.metrics.pinball_loss import pinball_loss
from src.metrics.coverage_accuracy import coverage_accuracy


def evaluate_regression(y_true, y_pred):
    """
    Computes evaluation metrics for regression tasks.
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.

    Returns:
        dict: A dictionary with evaluation metrics.
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(absolute_percentage_error(y_true, y_pred)),
        'Positive MAPE ': np.mean(absolute_percentage_positive_error(y_true, y_pred)),
        'Negative MAPE ': np.mean(absolute_percentage_negative_error(y_true, y_pred)),
        'MNE': np.mean(negative_error(y_true, y_pred)),
        'MPE': np.mean(positive_error(y_true, y_pred)),
        'Pinball Loss': pinball_loss(y_true, y_pred, 0.8),
        'Coverage Accuracy': coverage_accuracy(y_true, y_pred),
    }
    return metrics


def compute_median_80_percent_error(y_true, y_pred):
    """
    Computes the Median 80% Average Error by excluding the top and bottom 10% errors.
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    Returns:
        float: Median 80% Average Error.
    """
    absolute_errors = np.abs(y_true - y_pred)
    sorted_errors = np.sort(absolute_errors)

    lower_bound = int(0.1 * len(sorted_errors))
    upper_bound = int(0.9 * len(sorted_errors))

    middle_80_percent_errors = sorted_errors[lower_bound:upper_bound]
    median_80_error = np.median(middle_80_percent_errors)

    return median_80_error
