import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

class FullyConnectedBeforeResNet(pl.LightningModule):
    def __init__(self, input_size, resnet_input_channels=3, num_outputs=1, learning_rate=0.001):
        super(FullyConnectedBeforeResNet, self).__init__()
        
        self.learning_rate = learning_rate

        # Define fully connected layers with hardcoded dimensions before ResNet
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 256),  # Hardcoded to 256 units
            nn.ReLU(),
            nn.Linear(256, 128),         # Hardcoded to 128 units
            nn.ReLU(),
            nn.Linear(128, resnet_input_channels * 64 * 64),  # Reshape to feed into ResNet (hardcoded for 64x64 image size)
            nn.ReLU()
        )
        
        # Load a pretrained ResNet (e.g., ResNet18)
        self.resnet = models.resnet18(pretrained=True)

        # Modify the input layer of ResNet to match the dimensions after the fully connected layers
        self.resnet.conv1 = nn.Conv2d(resnet_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer for regression
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_outputs)

    def forward(self, x):
        # Pass through the fully connected layers first
        x = self.fc_layers(x)

        # Reshape the output from fully connected layers to feed into ResNet
        x = x.view(x.size(0), -1, 64, 64)  # Batch_size x Channels x Height x Width (64x64 is for ResNet input)

        # Pass through ResNet
        x = self.resnet(x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

# Example usage with PyTorch Lightning
model = FullyConnectedBeforeResNet(input_size=100, resnet_input_channels=3, num_outputs=1)

# Initialize PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=10, gpus=1 if torch.cuda.is_available() else 0)

# Dummy data for testing
x = torch.randn(8, 100)  # 8 samples, 100 features each
y = torch.randn(8, 1)    # 8 samples, 1 target each

# Dummy dataset and dataloader
train_dataset = torch.utils.data.TensorDataset(x, y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)

# Train the model
trainer.fit(model, train_loader)


class ResNetForRegression(pl.LightningModule):
    def __init__(self, input_channels=3, num_outputs=1, learning_rate=0.001):
        super(ResNetForRegression, self).__init__()
        
        self.learning_rate = learning_rate

        # Load a pretrained ResNet model (e.g., ResNet18)
        self.resnet = models.resnet18(pretrained=True)

        # Modify the input layer of ResNet if input_channels is different from the default (3 channels for RGB images)
        if input_channels != 3:
            self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace the final fully connected layer for regression
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_outputs)  # Set to 1 output for a single regression target

    def forward(self, x):
        # Pass the input through ResNet
        x = self.resnet(x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ResNetForTabularRegression(pl.LightningModule):
    def __init__(self, input_dim, learning_rate=1e-3):
        super(ResNetForTabularRegression, self).__init__()
        
        # Initialize with input dimension and learning rate
        self.learning_rate = learning_rate
        
        # Load a pre-trained ResNet and customize it for non-image data
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace initial convolution and pooling layers with dense layers for tabular data
        self.fc1 = nn.Linear(input_dim, 64)  # Adjust based on input dimension
        self.fc2 = nn.Linear(64, 64)
        
        # Replace ResNet layers with custom linear layers and skip connections
        self.resnet.layer1 = self._make_layer(64, 2)
        self.resnet.layer2 = self._make_layer(128, 2)
        self.resnet.layer3 = self._make_layer(256, 2)
        self.resnet.layer4 = self._make_layer(512, 2)
        
        # Final fully connected layer for regression output
        self.fc_out = nn.Linear(512, 1)  # Single output for regression
        
    def _make_layer(self, out_channels, blocks):
        # Define custom dense layers for residual blocks
        layers = []
        for _ in range(blocks):
            layers.append(nn.Linear(out_channels, out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Pass input through custom dense layers instead of convolutions
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Pass through the modified ResNet layers
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # Flatten and pass through the final fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc_out(x)  # No activation for regression output
        
        return x
    
    def training_step(self, batch, batch_idx):
        # Extract data and labels from batch
        x, y = batch
        # Forward pass
        y_hat = self(x)
        # Compute Mean Squared Error (MSE) loss
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Extract data and labels from batch
        x, y = batch
        # Forward pass
        y_hat = self(x)
        # Compute Mean Squared Error (MSE) loss
        val_loss = nn.functional.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        # Set up the optimizer
        return optim.Adam(self.parameters(), lr=self.learning_rate)
