import torch

from dl_lib.nn.activations import Tanh


def main() -> int:
    tanh = Tanh()
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    print("x =", x)
    print("tanh(x) =", tanh(x))

    return 0