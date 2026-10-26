import numpy as np


class SquareNet:
    def __init__(
        self,
        hidden_units: int = 4,
        low: float = -1.0,
        high: float = 1.0,
        seed: int | None = None
    ):
        rng = np.random.default_rng(seed=seed)
        self.hidden_weights = rng.uniform(low, high, size=hidden_units)
        self.hidden_bias = rng.uniform(low, high, size=hidden_units)
        self.output_weights = rng.uniform(low, high, size=hidden_units)
        self.output_bias = float(rng.uniform(low, high))