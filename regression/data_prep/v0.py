"""Data prep."""

import torch


def prepare_data(x_train, y_train):
    """Prepare data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train_tensor = torch.as_tensor(x_train, dtype=torch.float).to(device)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float).to(device)
    return (x_train_tensor, y_train_tensor)
