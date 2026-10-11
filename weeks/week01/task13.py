from typing import Literal

import numpy as np


def generate_samples(
    num_samples: int,
    val_range: tuple[float, float],
    distribution_type: Literal["uniform", "normal", "exponential"],
) -> np.ndarray:
    first, second = val_range
    if distribution_type == "uniform":
        return np.random.uniform(first, second, num_samples)
    if distribution_type == "normal":
        return np.random.normal(first, second, num_samples)
    if distribution_type == "exponential":
        return first + np.random.exponential(second, num_samples)

def main() -> None:
    num_samples = 10_000
    for distribution_type in ("uniform", "normal", "exponential"):
        samples = generate_samples(num_samples, (0.0, 1.0), distribution_type)
        print(
            f"{distribution_type.capitalize()}: "
            f"mean={np.mean(samples):.4f}, "
            f"std={np.std(samples):.4f}, "
            f"min={np.min(samples):.4f}, "
            f"max={np.max(samples):.4f}"
        )
