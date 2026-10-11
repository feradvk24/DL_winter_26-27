from typing import Literal

import matplotlib.pyplot as plt
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


def create_histograms(
    uniform_samples: np.ndarray,
    normal_samples: np.ndarray,
    exponential_samples: np.ndarray,
):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    distributions = (
        ("Uniform", uniform_samples),
        ("Normal", normal_samples),
        ("Exponential", exponential_samples),
    )
    for ax, (title, samples) in zip(axes, distributions):
        ax.hist(samples, bins=30)
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(True)

    figure.tight_layout()
    return figure, axes


def main() -> None:
    num_samples = 10_000
    distributions = []
    for distribution_type in ("uniform", "normal", "exponential"):
        samples = generate_samples(num_samples, (0.0, 1.0), distribution_type)
        distributions.append(samples)
        print(
            f"{distribution_type.capitalize()}: "
            f"mean={np.mean(samples):.4f}, "
            f"std={np.std(samples):.4f}, "
            f"min={np.min(samples):.4f}, "
            f"max={np.max(samples):.4f}"
        )

    create_histograms(*distributions)
    plt.show()
