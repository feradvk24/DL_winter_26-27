import numpy as np


def initialize_weights(low: float, high: float, size: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size)