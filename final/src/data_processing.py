import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from collections import Counter


def load_and_process_data(file_path, input_features, target_variables, categorical_columns=[], test_size=0.3):
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
    df.dropna(inplace=True)

    # Prepare the feature matrix (X)
    X = df[input_features].copy()

    # Separate the categorical columns for one-hot encoding
    categorical_data = X[categorical_columns]

    # Perform one-hot encoding for categorical columns
    onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
    categorical_encoded = pd.DataFrame(onehot_encoder.fit_transform(categorical_data),
                                       columns=onehot_encoder.get_feature_names_out(categorical_columns),
                                       index=X.index)
    
    # Drop original categorical columns and replace with one-hot encoded columns
    X = X.drop(columns=categorical_columns)
    X = pd.concat([X, categorical_encoded], axis=1)

    # Prepare target vectors based on the provided target variables
    y_targets = df[target_variables]  # Get targets as a 2D array

    # Split the dataset into train and test sets (80% train, 30% test+val)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_targets, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Outlier detection - courtesy of Yassine's https://www.kaggle.com/yassineghouzam/titanic-top-4-with-ensemble-modeling

def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 2 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers  