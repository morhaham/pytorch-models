"""Sequential model."""

import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# One layer - 1 input and output
model = nn.Sequential(nn.Linear(1, 1)).to(device)
# print(model.state_dict())

# Two layers
model2 = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 1)).to(device)
print(model2.state_dict())
