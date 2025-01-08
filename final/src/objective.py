import pytorch_lightning as pl
from functools import partial
from torch.utils.data import DataLoader


def objective(trial, train_dataset, model):
    lr = trial.suggest_float("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    lr_gamma = trial.suggest_float("lr_gamma", 0.1, 0.5)
    l2_reg = trial.suggest_float("l2_reg", 1e-6, 1e-2)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    trainer = pl.Trainer(max_epochs=10,
                         logger=False,
                         enable_progress_bar=False)
    trainer.fit(model, train_loader)
    return trainer.callback_metrics["train_loss"].item()