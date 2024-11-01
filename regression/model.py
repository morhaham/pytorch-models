"""Linear Regression Model."""

import torch.nn as nn


class ManualLinearRegression(nn.Module):
    """ManualLinearRegression defines all the methods required to run regression."""

    def __init__(self):
        """Initialize parameters."""
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x):
        """Forward pass."""
        return self.linear(x)
