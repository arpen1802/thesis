# data_loader.py
# -----------------------------------
# Script to load, process, and split the data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

def load_and_process_data(file_path, input_features, target_variables, categorical_columns):
    """
    Loads the data from a Parquet file, processes features (including one-hot encoding for categorical columns),
    and splits it into train and test sets for the specified targets.

    Args:
        file_path (str): Path to the Parquet file containing the dataset.
        input_features (list): List of feature column names to use as input.
        target_variables (list): List of target column names (1 or 2 targets).
        categorical_columns (list): List of categorical columns to one-hot encode.

    Returns:
        tuple: X_train, X_test, y_train (2D array), y_test (2D array)
    """
    # Load the dataset from the Parquet file
    df = pd.read_parquet(file_path)

    # Prepare the feature matrix (X)
    X = df[input_features].copy()

    # Separate the categorical columns for one-hot encoding
    categorical_data = X[categorical_columns]

    # Perform one-hot encoding for categorical columns
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')
    categorical_encoded = pd.DataFrame(onehot_encoder.fit_transform(categorical_data),
                                       columns=onehot_encoder.get_feature_names_out(categorical_columns),
                                       index=X.index)

    # Drop original categorical columns and replace with one-hot encoded columns
    X = X.drop(columns=categorical_columns)
    X = pd.concat([X, categorical_encoded], axis=1)

    # Prepare target vectors based on the provided target variables
    y_targets = df[target_variables].values  # Get targets as a 2D array

    # Split the dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_targets, test_size=0.2, random_state=42
    )

    # Optionally, apply scaling to numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    # Example usage
    input_features = [
        'c_realstartsoc', 'weekday_numerical', 'c_chargingmethod', 'is_weekend',
        'mean_consumption', 'mean_duration', 'mean_dep_time', 'c_chargingtype',
        'delta_soc_real', 'start_hour', 'weekday', 'latitude', 'longitude',
        'is_home_spot', 'is_location_one', 'plugin_duration_hr'
    ]
    categorical_columns = ['c_chargingmethod', 'c_chargingtype', 'weekday']  # Specify categorical columns
    target_variables = ['delta_soc_real', 'plugin_duration_hr']  # Use 1 or 2 targets

    file_path = 'your_dataset.parquet'
    X_train, X_test, y_train, y_test = load_and_process_data(
        file_path, input_features, target_variables, categorical_columns
    )

    # Print the shapes of the datasets
    print("Training set shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
