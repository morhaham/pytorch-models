import os
import sys

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

import numpy as np
import torch
from viz import plot_losses
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import torch.optim as optim
from pipeline import Pipeline
import matplotlib.pyplot as plt
from utils import data_generate, data_prepare, model_config

# Synthatic Data generation
x, y = data_generate()
# 

# Data preperation
train_loader, val_loader = data_prepare(x, y)
# 

# Model configuration
lr, model, optimizer, loss_fn = model_config()
#

sbs = Pipeline(model, loss_fn, optimizer)
sbs.set_loaders(train_loader, val_loader)
sbs.set_tensorboard('classy')

sbs.train(n_epochs=200)

print(model.state_dict())
print(sbs.total_epochs)

sbs.save_checkpoint('model_checkpoint.pth')

# plt = sbs.plot_losses()
# plt.show()

# Predictions
new_data = np.array([.5, .3, .7, 2, 7]).reshape(-1, 1)
print(f'Making predictions with the given years of experience: {new_data}')
predictions = sbs.predict(new_data)
print(f'The predicted salaries(thousands of dollars): {predictions}')
# 

