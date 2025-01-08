import pytorch_lightning as pl
import torch.nn as nn
import torch


class UserEmbeddingModel(pl.LightningModule):
    def __init__(self, num_users, embedding_dim, input_dim, dropout_prob=0.2,
                 lr=0.00001, lr_step_size=25, lr_gamma=0.5, l1_lambda=1e-5, l2_reg=1e-6):
        super(UserEmbeddingModel, self).__init__()

        self.save_hyperparameters()
        self.embedding = nn.Embedding(num_users, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim + input_dim, 32),
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

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        self.l1_lambda = l1_lambda
        self.l2_reg = l2_reg

    def forward(self, user_ids, X):
        user_ids = user_ids.long()
        user_embeds = self.embedding(user_ids)
        user_embeds = user_embeds.squeeze(dim=1)
        x = torch.cat([X, user_embeds], dim=1)
        output = self.fc_layers(x)
        return output

    def training_step(self, batch, batch_idx):
        X, user_ids, y = batch
        y_hat = self.forward(user_ids, X)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, user_ids, y = batch
        y_hat = self.forward(user_ids, X)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, user_ids, y = batch
        y_hat = self.forward(user_ids, X)
        loss = nn.MSELoss()(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=self.lr)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.l2_reg)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.lr_step_size,
                                                    gamma=self.lr_gamma)
        return [optimizer], [scheduler]