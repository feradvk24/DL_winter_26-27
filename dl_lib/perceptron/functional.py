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