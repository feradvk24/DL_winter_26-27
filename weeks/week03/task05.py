import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    gradient_descent_step,
    initialize_weights,
    numerical_gradient,
)


def main() -> None:
    dataset = create_dataset(6)
    x = np.array([x for x, _ in dataset], dtype=float)
    y_true = np.array([y for _, y in dataset], dtype=float)
    seeded_weights = initialize_weights(0, 10, size=1, seed=42)
    unseeded_weights = initialize_weights(0, 10, size=1)
    learning_rate = 0.001

    def loss_fn(current_weights: np.ndarray) -> float:
        predictions = current_weights[0] * x
        return mse(y_true, predictions)

    print(f"Seed 42, initial w={seeded_weights[0]:.6f}")
    print(f"No seed, initial w={unseeded_weights[0]:.6f}")

    for epoch in range(1, 501):
        seeded_gradient = numerical_gradient(loss_fn, seeded_weights)
        seeded_weights = gradient_descent_step(
            seeded_weights, seeded_gradient, learning_rate
        )

        unseeded_gradient = numerical_gradient(loss_fn, unseeded_weights)
        unseeded_weights = gradient_descent_step(
            unseeded_weights, unseeded_gradient, learning_rate
        )

    print(
        f"Seed 42 after 500 epochs: w={seeded_weights[0]:.6f}, "
        f"loss={loss_fn(seeded_weights):.6f} (target: w=2.0)"
    )
    print(
        f"No seed after 500 epochs: w={unseeded_weights[0]:.6f}, "
        f"loss={loss_fn(unseeded_weights):.6f} (target: w=2.0)"
    )
