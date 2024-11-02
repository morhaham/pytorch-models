"""Data prep."""

from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, TensorDataset

TrainLoader = Iterable[list[float]]


def prepare_data(x_train: torch.tensor, y_train: torch.tensor) -> TrainLoader:
    """Prepare data."""
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float)
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    return train_loader
