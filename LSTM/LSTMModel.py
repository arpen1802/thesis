import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

class LSTMModel(pl.LightningModule):
    def __init__(self, sequence_length, num_features, lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        super(LSTMModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_units, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        # Define a fully connected output layer
        self.fc = nn.Linear(lstm_units, 1)

    def forward(self, x):
        # Pass data through LSTM layers
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the last output in the sequence
        
        # Pass the last output through the fully connected layer
        output = self.fc(lstm_out)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)
