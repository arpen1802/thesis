import torch
import torch.nn as nn
import pytorch_lightning as pl


class EVChargingModel(pl.LightningModule):
    def __init__(self, input_dim, targets=["plugin_duration"]):
        """
        Args:
            input_dim (int): Number of input features.
            targets (list): List of target variables to predict. Options are ["plugin_duration", "delta_soc"].
        """
        super(EVChargingModel, self).__init__()
        self.targets = targets  # List of targets to predict
        
        # Define shared layers
        self.layer_1 = nn.Linear(input_dim, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        
        # Define output layers conditionally based on the targets
        if "plugin_duration" in self.targets:
            self.output_plugin_duration = nn.Linear(16, 1)
        if "delta_soc" in self.targets:
            self.output_delta_soc = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))

        outputs = {}
        if "plugin_duration" in self.targets:
            plugin_duration = torch.clamp(self.output_plugin_duration(x), 0, 24)  # Clamped between 0 and 24
            outputs["plugin_duration"] = plugin_duration
        if "delta_soc" in self.targets:
            delta_soc = torch.clamp(self.output_delta_soc(x), 0, 100)  # Clamped between 0 and 100
            outputs["delta_soc"] = delta_soc
        
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = 0

        # Calculate loss for each target and log it
        if "plugin_duration" in self.targets:
            loss_plugin = nn.MSELoss()(outputs["plugin_duration"], y["plugin_duration"])
            self.log('train_loss_plugin_duration', loss_plugin, on_step=True, on_epoch=True)
            loss += loss_plugin
        if "delta_soc" in self.targets:
            loss_delta = nn.MSELoss()(outputs["delta_soc"], y["delta_soc"])
            self.log('train_loss_delta_soc', loss_delta, on_step=True, on_epoch=True)
            loss += loss_delta

        # Log the total training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)
        loss = 0

        # Calculate validation loss for each target and log it
        if "plugin_duration" in self.targets:
            loss_plugin = nn.MSELoss()(outputs["plugin_duration"], y["plugin_duration"])
            self.log('val_loss_plugin_duration', loss_plugin, on_step=False, on_epoch=True)
            loss += loss_plugin
        if "delta_soc" in self.targets:
            loss_delta = nn.MSELoss()(outputs["delta_soc"], y["delta_soc"])
            self.log('val_loss_delta_soc', loss_delta, on_step=False, on_epoch=True)
            loss += loss_delta

        # Log the total validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
