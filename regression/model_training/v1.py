"""Model training."""

from collections.abc import Callable
from torch import tensor

Loss = float
PerformTrainFn = Callable[[tensor, tensor], Loss]


def train_model(x, y, train_step_fn) -> list[Loss]:
    """Train model."""
    n_epochs = 1000
    losses = []
    for epoch in range(n_epochs):
        loss = train_step_fn(x, y)
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
