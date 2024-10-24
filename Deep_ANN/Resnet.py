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
