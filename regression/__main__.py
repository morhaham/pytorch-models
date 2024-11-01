"""Run regression model."""

import torch
import numpy as np

from data_prep.v0 import prepare_data
from model_config.v0 import config_model
from model_training.v0 import train_model

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
val_idx = idx[int(N * 0.8):]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

(x_train_tensor, y_train_tensor) = prepare_data(x_train, y_train)
(model, optimizer, loss_fn) = config_model()
train_model(model, loss_fn, optimizer, x_train_tensor, y_train_tensor)

print(model.state_dict())
