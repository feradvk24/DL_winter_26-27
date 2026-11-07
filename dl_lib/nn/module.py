from abc import ABC, abstractmethod

import torch


class Module(ABC):
    training: bool = True

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __call__(self, x: torch.Tensor):
        self.forward(x)

    def train(self) -> 'Module':
        self.training = True
        return self

    def eval(self) -> 'Module':
        self.training = False
        return self