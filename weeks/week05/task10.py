import torch

from dl_lib.nn.activations import Sigmoid


def main() -> int:
    sigmoid = Sigmoid()
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    print("x =", x)
    print("sigmoid(x) =", sigmoid(x))

    return 0