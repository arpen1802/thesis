# data_loader.py
# -----------------------------------
# Script to load, process, and split the data

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_and_process_data(file_path, input_features, target_variables):
    """
    Loads the data from a Parquet file, processes features, and splits
    it into train and test sets for the specified targets.

    Args:
        file_path (str): Path to the Parquet file containing the dataset.
        input_features (list): List of feature column names to use as input.
        target_variables (list): List of target column names (1 or 2 targets).

    Returns:
        tuple: X_train, X_test, y_train (2D array), y_test (2D array)
    """
    # Load the dataset from the Parquet file
    df = pd.read_parquet(file_path)

    # Prepare the feature matrix (X)
    X = df[input_features].values

    # Prepare target vectors based on the provided target variables
    y_targets = np.column_stack([df[target].values for target in target_variables])

    # Split the dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_targets, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example usage
    input_features = [
        'c_realstartsoc', 'weekday_numerical', 'c_chargingmethod', 'is_weekend',
        'mean_consumption', 'mean_duration', 'mean_dep_time', 'c_chargingtype',
        'delta_soc_real', 'start_hour', 'weekday', 'latitude', 'longitude',
        'is_home_spot', 'is_location_one', 'plugin_duration_hr'
    ]
    target_variables = ['delta_soc_real', 'plugin_duration_hr']  # Use 1 or 2 targets

    file_path = 'your_dataset.parquet'
    X_train, X_test, y_train, y_test = load_and_process_data(file_path, input_features, target_variables)

    # Print the shapes of the datasets
    print("Training set shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
