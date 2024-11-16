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
from torch.utils.tensorboard import SummaryWriter

TheModel = nn.Sequential


def config_model(
    device, train_loader
) -> tuple[PerformTrainStepFn, PerformValStepFn, TheModel]:
    """Perform model configuration."""
    lr = 0.1
    torch.manual_seed(42)

    model = nn.Sequential(nn.Linear(1, 1)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    loss_fn = nn.MSELoss(reduction="mean")
    summary_writer = SummaryWriter("runs/linear_regression")
    x_dummy, y_dummy = next(iter(train_loader))
    summary_writer.add_graph(model, x_dummy.to(device))
    return (
        make_train_step_fn(model, loss_fn, optimizer),
        make_val_step_fn(model, loss_fn),
        model,
        optimizer,
        summary_writer,
    )
