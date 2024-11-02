"""Run regression model."""

import numpy as np
import torch

from data_prep import prepare_data
from model_config import config_model
from model_training import train_model, validate_model
from viz import plot_losses

device = "cuda" if torch.cuda.is_available() else "cpu"

true_b = 1
true_w = 2
N = 100

np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = 0.1 * np.random.randn(N, 1)
y = true_b + true_w * x + epsilon

train_loader, val_loader = prepare_data(x, y)

train_step_fn, val_step_fn, model = config_model(device)
losses, val_losses = train_model(
    train_loader, val_loader, train_step_fn, val_step_fn, device
)

print(model.state_dict())
plot_losses(losses, val_losses)
