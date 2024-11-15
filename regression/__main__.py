"""Run regression model."""

import sys

import numpy as np
import torch

from data_prep import prepare_data
from model_config import config_model, load_model_state
from model_training import train_model
from viz import plot_losses

args = sys.argv[1:]
should_load_model = "--load" in args

print(should_load_model)
device = "cuda" if torch.cuda.is_available() else "cpu"

true_b = 1
true_w = 2
N = 100

np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = 0.1 * np.random.randn(N, 1)
y = true_b + true_w * x + epsilon

train_loader, val_loader = prepare_data(x, y)

train_step_fn, val_step_fn, model, optimizer, summary_writer = config_model(
    device, train_loader
)

saved_epoch = 0
saved_losses = []
saved_val_losses = []
if should_load_model:
    saved_epoch, saved_losses, saved_val_losses = load_model_state(model, optimizer)

n_epochs = saved_epoch if saved_epoch > 0 else 200
losses = saved_losses if len(saved_losses) > 0 else []
val_losses = saved_val_losses if len(saved_val_losses) > 0 else []

losses, val_losses = train_model(
    train_loader,
    val_loader,
    train_step_fn,
    val_step_fn,
    device,
    summary_writer,
    n_epochs,
    saved_losses,
    saved_val_losses,
)

print(model.state_dict())
# plot_losses(losses, val_losses)
