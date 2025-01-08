import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(self, sequence_length, num_features, lstm_units=256,
                 dropout_rate=0.5, learning_rate=0.0001, lr_step_size=25,
                 lr_gamma=0.5, l1_lambda=1e-5, l2_reg=1e-6):
        super(LSTMModel, self).__init__()

        self.save_hyperparameters()
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.dropout = nn.Dropout(dropout_rate)
        self.l2_reg = l2_reg

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=lstm_units,
                            num_layers=1, batch_first=True, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(lstm_units)

        # Define a fully connected output layer
        self.fc = nn.Sequential(
            nn.Linear(self.lstm_units, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        norm_out = self.layer_norm(lstm_out)
        output = norm_out[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.l2_reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.lr_step_size,
                                                    gamma=self.lr_gamma)
        return [optimizer], [scheduler]
