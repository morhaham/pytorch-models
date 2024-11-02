"""Visualization helpers"""

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")
matplotlib.use("WebAgg")


def plot_losses(losses, val_losses):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(losses, label="Training Loss", c="b")
    plt.plot(val_losses, label="Validation Loss", c="r")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig
