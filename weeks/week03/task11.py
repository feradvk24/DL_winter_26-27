from functools import partial

import numpy as np

from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptron.datasets import xor_dataset
from dl_lib.perceptron.functional import gradient_descent_step, numerical_gradient
from dl_lib.perceptron.xor import Xor, forward


def set_parameters(model: Xor, params: np.ndarray) -> None:
    # The array holds 4 hidden weights, 2 hidden biases, 2 output weights, 1 bias.
    model.hidden_weights = params[:4].reshape(2, 2)
    model.hidden_bias = params[4:6]
    model.output_weights = params[6:8]
    model.output_bias = float(params[8])


def loss_given_parameters(
    model: Xor,
    dataset: list[tuple],
    targets: np.ndarray,
    candidate_params: np.ndarray,
) -> float:
    original = (
        model.hidden_weights,
        model.hidden_bias,
        model.output_weights,
        model.output_bias,
    )

    try:
        set_parameters(model, candidate_params)
        predictions = np.array([model.forward(x) for x, _ in dataset])
        return float(binary_cross_entropy(targets, predictions))
    finally:
        (
            model.hidden_weights,
            model.hidden_bias,
            model.output_weights,
            model.output_bias,
        ) = original


def main() -> None:
    dataset = xor_dataset()
    targets = np.array([y for _, y in dataset])

    for label, seed in (
        ("First attempt - seed=42", 42),
        ("Second attempt - seed=1 (a different random start)", 1),
    ):
        model = Xor(low=-1.0, high=1.0, seed=seed)
        params = np.concatenate([
            model.hidden_weights.ravel(),
            model.hidden_bias,
            model.output_weights,
            np.array([model.output_bias]),
        ])
        network_loss = partial(loss_given_parameters, model, dataset, targets)

        print(f"{label}:")
        print(f"  Initial loss: {network_loss(params):.6f}")

        for epoch in range(1, 10_001):
            gradient = numerical_gradient(network_loss, params)
            params = gradient_descent_step(params, gradient, learning_rate=2.0)
            set_parameters(model, params)

            if epoch % 2_000 == 0:
                print(f"  epoch {epoch:6d}: loss={network_loss(params):.6f}")

        for x, _ in dataset:
            prediction = forward(model, x[0], x[1])
            print(f"  XOR({x[0]:.0f}, {x[1]:.0f}) = {prediction:.6f}")
        print()
