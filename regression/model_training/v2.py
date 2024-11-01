"""Model training."""

from collections.abc import Callable
from torch import tensor
import numpy as np

Loss = float
PerformTrainFn = Callable[[tensor, tensor], Loss]


def train_model(train_loader, train_step_fn) -> list[Loss]:
    """Train model."""
    n_epochs = 1000
    losses = []
    for epoch in range(n_epochs):
        mini_batch_losses = []
        for x_batch, y_batch in train_loader:
            mini_batch_loss = train_step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)
        loss = np.mean(mini_batch_losses)
        losses.append(loss)
    return losses


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
