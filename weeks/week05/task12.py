import torch

from dl_lib.nn.activations import ReLU


def main() -> int:
    relu = ReLU()
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    print("x =", x)
    print("relu(x) =", relu(x))

    return 0
