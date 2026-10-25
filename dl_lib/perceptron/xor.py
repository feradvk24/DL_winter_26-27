import numpy as np


class Xor:
    def __init__(
        self,
        low: float = 0,
        high: float = 1,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.hidden_weights = rng.uniform(low, high, size=(2, 2))
        self.hidden_bias = rng.uniform(low, high, size=2)
        self.output_weights = rng.uniform(low, high, size=2)
        self.output_bias = float(rng.uniform(low, high))
