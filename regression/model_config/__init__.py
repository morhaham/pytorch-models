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


def save_model_state(model, n_epochs, optimizer, loss, val_loss):
    checkpoint = {
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": losses,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, "model_checkpoint.pth")


def load_model_state(model, optimizer):
    checkpoint = torch.load("model_checkpoint.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_sate_dict"])
    saved_epoch = checkpoint["epoch"]
    saved_losses = checkpoint["loss"]
    saved_val_losses = checkpoint["val_loss"]
    return saved_epoch, saved_losses, saved_val_losses
