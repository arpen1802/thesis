import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_regression_results(y_true, y_pred, savefig=False, name='fig'):
    """
    Plots regression results, including prediction vs true values and residuals.
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted target values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Fit')
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs Predicted")
    plt.legend()
    plt.grid()
    if savefig:
        plt.savefig(f"plots/{name}-scatter.png", dpi=300)
    plt.show()

    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, bins=30, color='blue', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Distribution")
    plt.grid()
    if savefig:
        plt.savefig(f"plots/{name}-residual.png", dpi=300)
    plt.show()


def plot_average_error_by_bins(y_true, y_pred, bin_size=1, savefig=False, name='fig'):
    """
    Computes the average error between y_pred and y_true grouped by integer bins of y_true.
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
    Returns:
        pd.DataFrame: A DataFrame with bins, count of values in each bin, and average error.
    """
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    bins = (np.floor(y_true) // bin_size).astype(int) * bin_size
    data = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'bin': bins
    })
    grouped = data.groupby('bin').apply(
        lambda group: pd.Series({
            'count': len(group),
            'avg_error': np.mean(group['y_pred'] - group['y_true'])
        })
    ).reset_index()

    grouped = grouped.sort_values(by='bin').reset_index(drop=True)
    plt.figure(figsize=(10, 6))
    plt.bar(grouped['bin'], grouped['avg_error'], width=bin_size,
            align='edge', color='skyblue', edgecolor='black')
    # sns.histplot(grouped, kde=True, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Bins (Integer Part of y_true)')
    plt.ylabel('Average Error')
    plt.title('Average Error by Energy Need in %')
    plt.xticks(grouped['bin'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"plots/{name}-average-err.png", dpi=300)
    plt.show()
    return grouped


def plot_prediction_distribution(y_true, y_pred, kde=True, savefig=False, name='fig'):
    """
    Plots the distribution of predictions (y_pred) and true values (y_true).
    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.
        bins (int): Number of bins for the histogram.
        kde (bool): Whether to overlay a Kernel Density Estimate (KDE).
    Returns:
        None: Displays the plot. 
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    plt.figure(figsize=(10, 6))
    # Plot histogram for y_true
    plt.hist(y_true, alpha=0.6, color='blue', edgecolor='black', label='True Values', density=True)
    # Plot histogram for y_pred
    plt.hist(y_pred, alpha=0.6, color='orange',edgecolor='black', label='Predicted Values', density=True)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of True Values and Predictions for Plugin Duration')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"plots/{name}-overlated-dist.png", dpi=300)
    plt.show()