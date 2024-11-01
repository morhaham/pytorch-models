"""Data prep."""

import torch
from custom_dataset import CustomDataset


def prepare_data(x_train, y_train) -> CustomDataset:
    """Prepare data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float).to(device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float).to(device)
    return CustomDataset(x_train_tensor, y_train_tensor)
