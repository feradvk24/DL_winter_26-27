import numpy as np
import torch

from dl_lib.activations.functional import sigmoid


def main() -> None:
    values = np.array([-10.0, 0.0, 10.0])
    print(sigmoid(values))

    x = np.linspace(-10, 10, 200)
    y = sigmoid(x)

    print("Output range:", y.min(), y.max())

    values_torch = torch.tensor(x)
    sigmoid_torch = torch.sigmoid(values_torch)
    print("Torch Sigmoid Output range:", sigmoid_torch.min(), sigmoid_torch.max())