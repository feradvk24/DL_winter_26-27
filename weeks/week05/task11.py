import torch

from dl_lib.nn.activations import Tanh


def main() -> None:
    tanh = Tanh()
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    print("x =", x)
    print("tanh(x) =", tanh(x))
