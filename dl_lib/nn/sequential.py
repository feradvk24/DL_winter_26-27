import torch

from dl_lib.nn.module import Module


class Sequential(Module):
    def __init__(self, *modules: Module):
        self.modules = [m for m in modules]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self.modules:
            x = module(x)

        return x

    def append(self, module: Module) -> None:
        self.modules.append(module)

    def extend(self, modules: list[Module]) -> None:
        self.modules.extend(modules)

    def insert(self, index: int, module: Module) -> None:
        self.modules.insert(index, module)

    def parameters(self) -> list[torch.Tensor]:
        return [
            parameter
            for module in self.modules
            for parameter in module.parameters()
        ]

    def train(self) -> "Sequential":
        super().train()
        for module in self.modules:
            module.train()
        return self

    def eval(self) -> "Sequential":
        super().eval()
        for module in self.modules:
            module.eval()
        return self
