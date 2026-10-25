from collections.abc import Callable

import numpy as np


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
    ) -> tuple[np.ndarray, float]:
    weight_gradient = np.zeros_like(weights, dtype=float)
    bias_gradient = 0.0
    current_bias = 0.0 if bias is None else bias

    for x, y in dataset:
        x = np.atleast_1d(x)

        prediction = np.dot(x, weights) + current_bias
        error = prediction - y

        weight_gradient += 2 * error * x
        bias_gradient += 2 * error

    n = len(dataset)

    return weight_gradient / n, float(bias_gradient / n)

def forward(
    x: np.ndarray,
    weights: np.ndarray,
    bias: float | None = None,
    activation: Callable | None = None,
) -> float:
    prediction = np.dot(x, weights)

    if bias is not None:
        prediction += bias

    if activation is not None:
        prediction = activation(prediction)

    return float(prediction)