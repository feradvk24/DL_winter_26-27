import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    analytical_gradient,
    initialize_weights,
    numerical_gradient,
)


def main() -> None:
    dataset = create_dataset(6)
    weights = initialize_weights(0, 10, size=1, seed=42)

    x = np.array([x for x, _ in dataset], dtype=float)
    y_true = np.array([y for _, y in dataset], dtype=float)

    def loss_fn(current_weights: np.ndarray) -> float:
        predictions = current_weights[0] * x
        return mse(y_true, predictions)

    numerical = numerical_gradient(loss_fn, weights)
    analytical, _ = analytical_gradient(dataset, weights)
    difference = np.abs(numerical - analytical)

    print(f"Loss: {loss_fn(weights):.6f}")
    print(f"Numerical gradient: {numerical[0]:.6f}")
    print(f"Analytical gradient: {analytical[0]:.6f}")
    print(f"Difference: {difference[0]:.8f}")