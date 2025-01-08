import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class UserEmbeddingDataset(Dataset):
    def __init__(self, X, user_ids, y):
        self.X = X
        self.user_ids = user_ids.long()
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.user_ids[idx], self.y[idx]