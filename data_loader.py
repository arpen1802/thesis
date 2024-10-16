import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_process_data(file_path):
    """
    Loads the data from a Parquet file, extracts features, and splits
    it into train and test sets for energy need and departure time targets.

    Args:
        file_path (str): Path to the Parquet file containing the dataset.

    Returns:
        dict: A dictionary containing train and test sets for features and targets.
    """
    # Load the dataset from the Parquet file
    df = pd.read_parquet(file_path)

    # Extract datetime-based features
    if 'start_time' in df.columns:
        df['hour'] = df['start_time'].dt.hour  # Extract hour of the day
        df['day_of_week'] = df['start_time'].dt.dayofweek  # Monday=0, Sunday=6

    # Prepare the feature matrix (X) and target vectors (y)
    input_features = ['plugin_duration', 'delta_soc', 'hour', 'day_of_week']
    X = df[input_features].values

    # Define the target variables
    y_energy = df['energy_need'].values
    y_departure = df['departure_time'].values

    # Split the dataset into train and test sets (80% train, 20% test)
    X_train, X_test, y_energy_train, y_energy_test = train_test_split(
        X, y_energy, test_size=0.2, random_state=42
    )

    _, _, y_departure_train, y_departure_test = train_test_split(
        X, y_departure, test_size=0.2, random_state=42
    )

    # Store the splits in a dictionary for easy access
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_energy_train': y_energy_train,
        'y_energy_test': y_energy_test,
        'y_departure_train': y_departure_train,
        'y_departure_test': y_departure_test
    }

    return data_dict

if __name__ == '__main__':
    # Example usage
    file_path = 'your_dataset.parquet'
    data = load_and_process_data(file_path)

    # Print the shapes of the datasets
    print("Training set shapes:")
    print(f"X_train: {data['X_train'].shape}, y_energy_train: {data['y_energy_train'].shape}")
    print(f"y_departure_train: {data['y_departure_train'].shape}")
