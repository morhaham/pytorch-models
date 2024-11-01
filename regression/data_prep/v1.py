"""Data prep."""

import torch
from collections.abc import Iterable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def prepare_data(x_train: torch.tensor, y_train: torch.tensor) -> Iterable[list[float]]:
    """Prepare data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float).to(device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float).to(device)
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=16,
        shuffle=True
    )
    return train_loader
