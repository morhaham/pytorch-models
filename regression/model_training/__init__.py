"""Model training."""

from collections.abc import Callable

import numpy as np
from data_prep.v1 import TrainLoader
from torch import tensor

Loss = float
PerformTrainFn = Callable[[tensor, tensor], Loss]


def train_model(train_loader: TrainLoader, train_step_fn, device) -> list[Loss]:
    """Train model."""
    n_epochs = 200
    losses = []
    for epoch in range(n_epochs):
        loss = mini_batch(train_loader, train_step_fn, device)
        losses.append(loss)
    return losses


def mini_batch(train_loader, train_step_fn, device):
    mini_batch_losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        mini_batch_loss = train_step_fn(x_batch, y_batch)
        mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
    return loss


def make_train_step_fn(model, loss_fn, optimizer) -> PerformTrainFn:
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
