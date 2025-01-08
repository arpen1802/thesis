import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset


class EVChargingModel(pl.LightningModule):
    def __init__(self, input_dim, targets=["plugin_duration"], dropout_prob = 0.2, l1_lambda=1e-5):
        """
        Args:
            input_dim (int): Number of input features.
            targets (list): List of target variables to predict. Options are ["plugin_duration", "delta_soc"].
        """
        super(EVChargingModel, self).__init__()
        self.targets = targets  # List of targets to predict
        self.l1_lambda = l1_lambda

        # Define shared layers
        self.layer_1 = nn.Linear(input_dim, 512)
        self.layer_2 = nn.Linear(512, 256)
        self.layer_3 = nn.Linear(256, 128)
        self.layer_4 = nn.Linear(128, 64)
        self.layer_5 = nn.Linear(64, 32)
        self.layer_6 = nn.Linear(32, 16)
        self.dropout = nn.Dropout(p=dropout_prob)

        # Define output layers conditionally based on the targets
        if "plugin_duration" in self.targets:
            self.output_plugin_duration = nn.Linear(16, 1)
        if "delta_soc" in self.targets:
            self.output_delta_soc = nn.Linear(16, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.relu(self.layer_3(x))
        x = self.dropout(x)
        x = self.relu(self.layer_4(x))
        x = self.dropout(x)
        x = self.relu(self.layer_5(x))
        x = self.dropout(x)
        x = self.relu(self.layer_6(x))
        x = self.dropout(x)

        outputs = {}
        if "plugin_duration" in self.targets:
            # raw_plugin_duration = self.output_plugin_duration(x)
            plugin_duration = self.output_plugin_duration(x) # Clamped between 0 and 24
            plugin_duration = torch.sigmoid(plugin_duration) * 24
            outputs["plugin_duration"] = plugin_duration
        if "delta_soc" in self.targets:
            delta_soc = self.output_delta_soc(x) # Clamped between 0 and 100
            delta_soc = torch.sigmoid(delta_soc) * 100
            outputs["delta_soc"] = delta_soc

        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = 0

        # Calculate loss for each target and log it
        if "plugin_duration" in self.targets:
            loss_plugin = nn.MSELoss()(outputs["plugin_duration"], y[:, 0])
            self.log('train_loss_plugin_duration', loss_plugin, on_step=False, on_epoch=True)
            loss += loss_plugin
        if "delta_soc" in self.targets:
            loss_delta = nn.MSELoss()(outputs["delta_soc"], y[:, 1])
            self.log('train_loss_delta_soc', loss_delta, on_step=False, on_epoch=True)
            loss += loss_delta

        # Add L1 regularization
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        loss += self.l1_lambda * l1_norm
        
        # Log the total training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = 0

        # Calculate validation loss for each target and log it
        if "plugin_duration" in self.targets:
            loss_plugin = nn.MSELoss()(outputs["plugin_duration"], y[:, 0])
            self.log('val_loss_plugin_duration', loss_plugin, on_step=False, on_epoch=True)
            loss += loss_plugin
        if "delta_soc" in self.targets:
            loss_delta = nn.MSELoss()(outputs["delta_soc"], y[:, 1])
            self.log('val_loss_delta_soc', loss_delta, on_step=False, on_epoch=True)
            loss += loss_delta

        # Log the total validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


# Custom dataset class
class EVDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]