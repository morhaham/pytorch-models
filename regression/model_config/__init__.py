"""Model config."""

import torch
import torch.nn as nn
import torch.optim as optim
from model_training import (
    PerformTrainStepFn,
    PerformValStepFn,
    make_train_step_fn,
    make_val_step_fn,
)

TheModel = nn.Sequential


def config_model(device) -> tuple[PerformTrainStepFn, PerformValStepFn, TheModel]:
    """Perform model configuration."""
    lr = 0.1
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(1, 1)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="mean")
    return (
        make_train_step_fn(model, loss_fn, optimizer),
        make_val_step_fn(model, loss_fn),
        model,
    )
