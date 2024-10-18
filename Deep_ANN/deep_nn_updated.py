import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Assuming load_and_process_data is available from data_loader.py
from data_loader import load_and_process_data

# Custom dataset class
class EVDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# Define the deep ANN model
class EVChargingModel(pl.LightningModule):
    def __init__(self, input_dim):
        super(EVChargingModel, self).__init__()
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.output(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Load the data using load_and_process_data function
file_path = 'your_dataset.parquet'
input_features = [
    'c_realstartsoc', 'weekday_numerical', 'c_chargingmethod', 'is_weekend',
    'mean_consumption', 'mean_duration', 'mean_dep_time', 'c_chargingtype',
    'delta_soc_real', 'start_hour', 'weekday', 'latitude', 'longitude',
    'is_home_spot', 'is_location_one', 'plugin_duration_hr'
]
target_variables = ['delta_soc_real']  # Assuming you're predicting energy need

X_train, X_test, y_energy_train, y_energy_test = load_and_process_data(
    file_path=file_path,
    input_features=input_features,
    target_variables=target_variables
)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_energy_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_energy_test, dtype=torch.float32).view(-1, 1)

# Create datasets and dataloaders
train_dataset = EVDataset(X_train_tensor, y_train_tensor)
val_dataset = EVDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model and trainer
model = EVChargingModel(input_dim=X_train_tensor.shape[1])
trainer = pl.Trainer(max_epochs=15)

# Train the model
trainer.fit(model, train_loader, val_loader)
