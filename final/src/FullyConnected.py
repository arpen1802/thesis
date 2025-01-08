import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl


class FullyConnected(pl.LightningModule):
    def __init__(self, input_size, dropout_prob=0.2, learning_rate=0.001,
                 lr_step_size=50, lr_gamma=0.28, l1_lambda=1e-5, l2_reg=1e-6):
        super(FullyConnected, self).__init__()

        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        self.l1_lambda = l1_lambda
        self.l2_reg = l2_reg

        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        output = self.fc_layers(x)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.lr_step_size,
                                                    gamma=self.lr_gamma)
        return [optimizer], [scheduler]


