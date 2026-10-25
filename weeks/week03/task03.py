import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import (
    gradient_descent_step,
    initialize_weights,
    numerical_gradient,
)


def main() -> None:
    data = create_dataset(6)
    x = np.array([sample[0] for sample in data], dtype=float)
    y_true = np.array([sample[1] for sample in data], dtype=float)
    initial_weights = initialize_weights(0, 10, size=1, seed=42)
    learning_rate = 0.001

    def loss_fn(params: np.ndarray) -> float:
        y_pred = params[0] * x
        return mse(y_true, y_pred)

    gradient = numerical_gradient(loss_fn, initial_weights)
    raw_weights = initial_weights - gradient
    scaled_weights = gradient_descent_step(initial_weights, gradient, learning_rate)
    
    print(f"Initial w: {initial_weights[0]:.6f}, loss: {loss_fn(initial_weights):.6f}")
    print(f"Numerical gradient: {gradient[0]:.6f}")
    print(f"w={raw_weights[0]:.6f}, loss={loss_fn(raw_weights):.6f}")
    print(
        f"w after one step with learning_rate={learning_rate}: "
        f"w={scaled_weights[0]:.6f}, loss={loss_fn(scaled_weights):.6f}"
    )
    
    weights = initial_weights.copy()
    print(f"Loss before update: {loss_fn(weights):.6f}")
    for epoch in range(1, 11):
        gradient = numerical_gradient(loss_fn, weights)
        weights = gradient_descent_step(weights, gradient, learning_rate)
        print(f"epoch {epoch:2d}: w={weights[0]:.6f}, loss={loss_fn(weights):.6f}")
