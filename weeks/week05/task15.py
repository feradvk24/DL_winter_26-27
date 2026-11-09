import torch

from dl_lib.nn.activations import ReLU
from dl_lib.nn.module import Module
from dl_lib.nn.sequential import Sequential


class _Affine(Module):
    def __init__(self, n_inputs: int, n_outputs: int):
        self.weight = torch.randn(n_inputs, n_outputs, requires_grad=True)
        self.bias = torch.zeros(n_outputs, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight + self.bias

    def parameters(self) -> list[torch.Tensor]:
        return [self.weight, self.bias]


def main() -> int:
    sequential = Sequential(_Affine(3, 4), ReLU(), _Affine(4, 2))
    parameters = sequential.parameters()

    print("number of learnable tensors =", len(parameters))
    for index, parameter in enumerate(parameters):
        print(f"params[{index}].shape = {tuple(parameter.shape)}")

    return 0
