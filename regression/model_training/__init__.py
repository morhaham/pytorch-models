"""Model training."""

from collections.abc import Callable

import numpy as np
from torch import tensor, torch
from torch.utils.data import DataLoader

Loss = float
PerformTrainStepFn = PerformValStepFn = Callable[[tensor, tensor], Loss]


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_step_fn,
    val_step_fn,
    device,
    summary_writer,
    n_epochs,
    losses,
    val_losses,
) -> list[Loss]:
    """Train model."""
    for epoch in range(n_epochs):
        loss = mini_batch(train_loader, train_step_fn, device)
        losses.append(loss)
        with torch.no_grad():
            val_loss = mini_batch(val_loader, val_step_fn, device)
            val_losses.append(val_loss)

        summary_writer.add_scalars(
            main_tag="loss",
            tag_scalar_dict={"training": loss, "validation": val_loss},
            global_step=epoch,
        )
    summary_writer.close()
    return losses, val_losses


def mini_batch(data_loader, step_fn, device):
    mini_batch_losses = []
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
    loss = np.mean(mini_batch_losses)
    return loss


def make_train_step_fn(model, loss_fn, optimizer) -> PerformTrainStepFn:
    """Perform train step."""

    def perform_train_step_fn(x, y):
        model.train()
        yhat = model(x)

        loss = loss_fn(yhat, y)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    return perform_train_step_fn


def make_val_step_fn(model, loss_fn) -> PerformTrainStepFn:
    """Perform validation step."""

    def perform_val_step_fn(x, y):
        model.eval()
        yhat = model(x)

        loss = loss_fn(yhat, y)
        return loss.item()

    return perform_val_step_fn


def save_model_state(model, n_epochs, optimizer, losses, val_losses):
    checkpoint = {
        "epoch": n_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "val_losses": val_losses,
    }
    torch.save(checkpoint, "model_checkpoint.pth")


def load_model_state(model, optimizer):
    checkpoint = torch.load("model_checkpoint.pth")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    saved_epoch = checkpoint["epoch"]
    saved_losses = checkpoint["losses"]
    saved_val_losses = checkpoint["val_losses"]
    return saved_epoch, saved_losses, saved_val_losses
