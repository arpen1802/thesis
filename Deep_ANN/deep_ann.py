import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

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
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

# Assuming the data has been preprocessed as mentioned in the paper
# Load and preprocess the data (dummy example)
data = pd.read_csv('/path/to/your/data.csv')
target = data.pop('target_variable')  # Replace with the actual target variable column name
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Convert to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
target = torch.tensor(target.values, dtype=torch.float32).view(-1, 1)

# Create datasets and dataloaders
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(EVDataset(data, target), [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Initialize model, trainer, and train the model
model = EVChargingModel(input_dim=data.shape[1])
trainer = pl.Trainer(max_epochs=15)
trainer.fit(model, train_loader, val_loader)
