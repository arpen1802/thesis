import torch
import torch.nn as nn
import pytorch_lightning as pl


class UserEmbeddingLstm(pl.LightningModule):
    def __init__(self, sequence_length, num_features, num_users, embedding_dim,
                 lstm_units=512, dropout_rate=0.5, learning_rate=0.00005,
                 lr_step_size=25, lr_gamma=0.5, l1_lambda=1e-5, l2_reg=1e-6):
        super(UserEmbeddingLstm, self).__init__()

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

        self.embedding = nn.Embedding(num_users, embedding_dim)

        # Define LSTM layers
        self.lstm = nn.LSTM(input_size=embedding_dim + num_features, hidden_size=lstm_units,
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

    def forward(self, user_ids, X):
        user_ids = user_ids.long()
        user_embeds = self.embedding(user_ids)

        # Expand user embeddings to match sequence length
        # Repeat embeddings along sequence_length dimension: (batch_size, sequence_length, embedding_dim)
        user_embeds_expanded = user_embeds.unsqueeze(1).repeat(1, X.size(1), 1)

        # Concatenate embeddings with sequence data
        # Concatenated shape: (batch_size, sequence_length, input_dim + embedding_dim)
        lstm_input = torch.cat([X, user_embeds_expanded], dim=2)
        lstm_out, _ = self.lstm(lstm_input)

        norm_out = self.layer_norm(lstm_out)
        last_hidden_state = norm_out[:, -1, :]

        output = self.dropout(last_hidden_state)
        output = self.fc(output)
        return output

    def training_step(self, batch, batch_idx):
        x, user_ids, y = batch
        y_hat = self(user_ids, x)
        loss = nn.L1Loss()(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, user_ids, y = batch
        y_hat = self(user_ids, x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, user_ids, y = batch
        y_hat = self(user_ids, x)
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
