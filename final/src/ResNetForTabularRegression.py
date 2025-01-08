import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl


class ResNetForTabularRegression(pl.LightningModule):
    def __init__(self, input_dim, learning_rate=1e-3):
        super(ResNetForTabularRegression, self).__init__()
        
        # Initialize with input dimension and learning rate
        self.learning_rate = learning_rate
        
        # Load a pre-trained ResNet and customize it for non-image data
        self.resnet = models.resnet18(weights=None)
        
        # Replace initial convolution and pooling layers with dense layers for tabular data
        self.fc1 = nn.Linear(input_dim, 64)  # Adjust based on input dimension
        self.fc2 = nn.Linear(64, 64)
        
        # Replace ResNet layers with custom linear layers and skip connections
        # self.resnet.layer1 = self._make_layer(64, 2)
        # self.resnet.layer2 = self._make_layer(128, 2)
        # self.resnet.layer3 = self._make_layer(256, 2)
        # self.resnet.layer4 = self._make_layer(512, 2)
        self.fc_in = nn.Linear(input_dim, 512)
        
        # Final fully connected layer for regression output
        self.resnet.fc = nn.Linear(512, 1)  # Single output for regression
        
    # def _make_layer(self, out_channels, blocks):
    #     # Define custom dense layers for residual blocks
    #     layers = []
    #     for _ in range(blocks):
    #         layers.append(nn.Linear(out_channels, out_channels))
    #         layers.append(nn.ReLU(inplace=True))
    #     return nn.Sequential(*layers)
    
    def forward(self, x):
        # Pass input through custom dense layers instead of convolutions
        x = torch.relu(self.fc_in(x))
        # x = torch.relu(self.fc2(x))
        
        # # Pass through the modified ResNet layers
        # x = self.resnet.layer1(x)
        # x = self.resnet.layer2(x)
        # x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        x = x.view(x.size(0), 512, 1, 1)
        
        # Flatten and pass through the final fully connected layer
        # x = torch.flatten(x, 1)
        x = self.resnet(x)  # No activation for regression output
        
        return x
    
    def training_step(self, batch, batch_idx):
        # Extract data and labels from batch
        x, y = batch
        # Forward pass
        y_hat = self(x)
        # Compute Mean Squared Error (MSE) loss
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract data and labels from batch
        x, y = batch
        # Forward pass
        y_hat = self(x)
        # Compute Mean Squared Error (MSE) loss
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        return val_loss

    def configure_optimizers(self):
        # Set up the optimizer
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)