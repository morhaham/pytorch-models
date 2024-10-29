"""Linear Regression Model."""

import torch
import torch.nn as nn


class ManualLinearRegression(nn.Module):
    """ManualLinearRegression defines all the methods required to run regression."""

    def __init__(self):
        """Initialize parameters."""
        super().__init__()
        self.b = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x):
        """Forward pass."""
        return self.b + self.w * x
