from collections.abc import Callable

import numpy as np

from dl_lib.perceptron.functional import forward, initialize_weights


class Perceptron:
    def __init__(
        self,
        size: int,
        low: float = 0,
        high: float = 10,
        seed: int | None = None,
        use_bias: bool = False,
        activation: Callable | None = None
        ):
        self.weights = initialize_weights(low=low, high=high, size=size, seed=seed)
        self.use_bias = use_bias
        self.activation = activation
        self.bias = 0.0
        if use_bias:
            rng = np.random.default_rng(seed)
            rng.uniform(low, high)

    def forward(self, n_inputs):
        return forward(n_inputs, self.weights, self.bias, self.activation)

