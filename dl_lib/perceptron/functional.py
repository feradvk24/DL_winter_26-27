from collections.abc import Callable

import numpy as np

from dl_lib.losses.functional import mse


def initialize_weights(low: float, high: float, size: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=size)


def numerical_gradient(loss_fn: Callable, params: np.ndarray):
    h = 1e-5
    gradient = np.zeros_like(params, dtype=float)

    for i in np.ndindex(params.shape):
        plus = params.astype(float, copy=True)
        minus = params.astype(float, copy=True)

        plus[i] += h
        minus[i] -= h

        gradient[i] = (loss_fn(plus) - loss_fn(minus)) / (2 * h)

    return gradient

def gradient_descent_step(params: np.ndarray, gradient: np.ndarray, learning_rate: float) -> np.ndarray:
    return params - learning_rate * gradient

def analytical_gradient(
    dataset: list[tuple],
    weights: np.ndarray,
    bias:  float | None = None,
    activation: Callable | None = None,
    activation_derivative: Callable | None = None,
    loss: str = "mse",
    ) -> tuple[np.ndarray, float]:
    weight_gradient = np.zeros_like(weights, dtype=float)
    bias_gradient = 0.0
    current_bias = 0.0 if bias is None else bias

    for x, y in dataset:
        x = np.atleast_1d(x)
        z = forward(x, weights, current_bias)

        prediction = z
        if activation is not None:
            prediction = activation(z)

        error = prediction - y

        if loss == "bce":
            delta = error
        
        elif loss == "mse":
            delta = 2 * error

            if activation is not None:
                if activation_derivative is None:
                    raise ValueError("MSE with activation needs derivative")

                delta *= activation_derivative(z)

        else:
            raise ValueError(f"Unsupported loss: {loss}")

        weight_gradient += delta * x
        bias_gradient += delta

    n = len(dataset)

    return weight_gradient / n, float(bias_gradient / n)

def forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    activation: Callable | None = None,
) -> float:
    prediction = np.dot(x, weights)
    prediction += bias

    if activation is not None:
        prediction = activation(prediction)

    return float(prediction)

def dataset_loss(
    dataset: list[tuple],
    targets: np.ndarray,
    weights: np.ndarray,
    bias: float = 0.0,
    activation: Callable | None = None,
    loss: Callable = mse,
) -> float:
    predictions = np.array([
        forward(x, weights, bias, activation)
        for x, _ in dataset
    ])

    return float(loss(targets, predictions))

def dataset_loss_given_bias(
    dataset: list[tuple],
    targets: np.ndarray,
    weights: np.ndarray,
    activation: Callable | None,
    bias: np.ndarray,
    loss: Callable = mse,
) -> float:
    return dataset_loss(
        dataset,
        targets,
        weights,
        float(bias.item()),
        activation,
        loss=loss,
    )
