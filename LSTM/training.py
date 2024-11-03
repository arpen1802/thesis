import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTMModel

# Load your data
# Assume 'df' is a DataFrame with your dataset, filtered and sorted by user and 'start_date'
# Prepare the dataset
sequence_length = 5
features = ['delta_soc', 'other_features']
target = 'plugin_duration'

# Normalize features
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Generate sequences
sequences = []
targets = []
for i in range(len(df) - sequence_length):
    sequences.append(df[features].iloc[i:i + sequence_length].values)
    targets.append(df[target].iloc[i + sequence_length])

X = np.array(sequences)
y = np.array(targets)

# Split data
train_size = int(0.8 * len(X))
val_size = int(0.1 * len(X))

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Instantiate the model
model = LSTMModel(sequence_length=sequence_length, num_features=len(features))

# Define early stopping
early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

# Initialize the Trainer
trainer = pl.Trainer(max_epochs=50, callbacks=[early_stopping])

# Train the model
trainer.fit(model, train_loader, val_loader)

# Test the model
trainer.test(model, test_loader)
