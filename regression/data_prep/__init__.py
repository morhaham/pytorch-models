"""Data prep."""

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


def prepare_data(x: torch.tensor, y: torch.tensor) -> tuple[DataLoader, DataLoader]:
    """Prepare data."""
    torch.manual_seed(13)
    x_tensor = torch.as_tensor(x, dtype=torch.float)
    y_tensor = torch.as_tensor(y, dtype=torch.float)
    dataset = TensorDataset(x_tensor, y_tensor)
    ratio = 0.8
    n_total = len(dataset)
    n_train = int(n_total * ratio)
    n_val = n_total - n_train
    train_data, val_data = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=16)
    return train_loader, val_loader
