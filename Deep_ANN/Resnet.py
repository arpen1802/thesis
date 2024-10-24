import torch
import torch.nn as nn
import torchvision.models as models

class FullyConnectedBeforeResNet(nn.Module):
    def __init__(self, input_size, resnet_input_channels=3, num_outputs=1):
        super(FullyConnectedBeforeResNet, self).__init__()
        
        # Define hardcoded fully connected layers before ResNet
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 256),  # First fully connected layer (hardcoded to 256 units)
            nn.ReLU(),
            nn.Linear(256, 128),         # Second fully connected layer (hardcoded to 128 units)
            nn.ReLU(),
            nn.Linear(128, resnet_input_channels * 64 * 64),  # Final layer before ResNet input
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

# Example usage
model = FullyConnectedBeforeResNet(input_size=100, resnet_input_channels=3, num_outputs=1)

# Example input (batch size = 8, input size = 100)
x = torch.randn(8, 100)  # Adjust input size based on your actual input dimensions
output = model(x)

# Print output shape (should be [8, 1] for regression)
print(output.shape)
