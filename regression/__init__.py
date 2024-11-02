"""Run regression model."""

from model import ManualLinearRegression
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

true_b = 1
true_w = 2
N = 100

np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = 0.1 * np.random.randn(N, 1)
y = true_b + true_w * x + epsilon

idx = np.arange(N)
np.random.shuffle(idx)

train_idx = idx[: int(N * 0.8)]
val_idx = idx[int(N * 0.8) :]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]
x_train_tensor = torch.as_tensor(x_train).to(device)
y_train_tensor = torch.as_tensor(y_train).to(device)

lr = 0.1
torch.manual_seed(42)
model = ManualLinearRegression().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

loss_fn = nn.MSELoss(reduction="mean")
n_epochs = 1000

for epoch in range(n_epochs):
    model.train()
    yhat = model(x_train_tensor)

    loss = loss_fn(yhat, y_train_tensor)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

print(model.state_dict())
print(list(model.parameters()))
