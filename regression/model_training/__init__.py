"""Model training."""

from collections.abc import Callable

import numpy as np
from torch import tensor, torch
from torch.utils.data import DataLoader

Loss = float
PerformTrainStepFn = PerformValStepFn = Callable[[tensor, tensor], Loss]


def train_model(
    train_loader: DataLoader, val_loader: DataLoader, train_step_fn, val_step_fn, device
) -> list[Loss]:
    """Train model."""
    n_epochs = 200
    losses = []
    val_losses = []
    for epoch in range(n_epochs):
        loss = mini_batch(train_loader, train_step_fn, device)
        losses.append(loss)
        with torch.no_grad():
            loss = mini_batch(val_loader, val_step_fn, device)
            val_losses.append(loss)
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
