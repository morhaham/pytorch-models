"""Custom Dataset."""

from torch.utils.data import Dataset, TensorDataset, DataLoader


class CustomDataset(Dataset):
    """Custom Dataset."""

    def __init__(self, x_tensor, y_tensor):
        """Initialize new dataset."""
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        """Get a tuple of (features, label)."""
        return (self.x[index], self.y[index])

    def __len__(self):
        """Get the features length."""
        return len(self.x)
