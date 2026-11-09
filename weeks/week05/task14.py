import torch

from dl_lib.nn.activations import ReLU, Tanh
from dl_lib.nn.sequential import Sequential


def main() -> int:
    sequential = Sequential(ReLU(), Tanh())
    x = torch.tensor([-2.0, 2.0])

    print("x =", x)
    print("sequential(x) =", sequential(x))

    return 0