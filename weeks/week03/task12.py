from functools import partial

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.functional import (
    gradient_descent_step,
    numerical_gradient,
)
from dl_lib.perceptron.square_net import SquareNet


def set_parameters(model: SquareNet, params: np.ndarray) -> None:
    hidden_units = model.hidden_weights.size

    model.hidden_weights = params[:hidden_units]
    model.hidden_bias = params[hidden_units:2 * hidden_units]
    model.output_weights = params[2 * hidden_units:3 * hidden_units]
    model.output_bias = float(params[-1])


def loss_given_parameters(
    model: SquareNet,
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

        predictions = np.array([
            model.forward(x)
            for x, _ in dataset
        ])

        return float(mse(targets, predictions))
    finally:
        (
            model.hidden_weights,
            model.hidden_bias,
            model.output_weights,
            model.output_bias,
        ) = original


def main() -> None:
    dataset = [
        (np.array([float(x)]), float(x**2))
        for x in range(-8, 9)
    ]
    targets = np.array([y for _, y in dataset])

    model = SquareNet(
        hidden_units=12,
        low=-1.0,
        high=1.0,
        seed=42,
    )

    predictions = np.array([
        model.forward(x)
        for x, _ in dataset
    ])

    initial_loss = mse(targets, predictions)
    print("Initial loss:", initial_loss)

    params = np.concatenate([
        model.hidden_weights,
        model.hidden_bias,
        model.output_weights,
        np.array([model.output_bias]),
    ])

    network_loss = partial(
        loss_given_parameters,
        model,
        dataset,
        targets,
    )

    learning_rate = 0.01
    epochs = 80_000

    print("Training:")

    for epoch in range(1, epochs + 1):
        gradient = numerical_gradient(network_loss, params)

        params = gradient_descent_step(
            params,
            gradient,
            learning_rate,
        )

        set_parameters(model, params)

        if epoch % 10_000 == 0:
            current_loss = network_loss(params)
            print(f"  epoch {epoch:6d}: loss={current_loss:.6f}")

    print("\nPredictions:")

    for value in [-8, -7, -2, 0, 2, 7, 8]:
        prediction = model.forward(np.array([float(value)]))

        print(
            f"  model({value}) = {prediction:.3f} "
            f"(x^2 = {value**2})"
        )

    prediction_2 = model.forward(np.array([2.0]))
    prediction_7 = model.forward(np.array([7.0]))

    np.testing.assert_allclose(
        prediction_2, 4.0, atol=0.5, rtol=0
    )
    np.testing.assert_allclose(
        prediction_7, 49.0, atol=0.5, rtol=0
    )