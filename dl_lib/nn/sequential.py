import torch

from dl_lib.nn.module import Module


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = [m for m in modules]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules:
            x = module(x)

        return x