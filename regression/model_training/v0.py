"""Model training."""


def train_model(model, loss_fn, optimizer, x_train_tensor, y_train_tensor):
    """Train model."""
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        yhat = model(x_train_tensor)

        loss = loss_fn(yhat, y_train_tensor)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
