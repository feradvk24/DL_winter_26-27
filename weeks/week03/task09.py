from functools import partial

import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptron.datasets import and_dataset
from dl_lib.perceptron.functional import (
    analytical_gradient,
    dataset_loss,
    dataset_loss_given_bias,
    numerical_gradient,
)
from dl_lib.perceptron.perceptron import Perceptron


def main() -> None:
    dataset = and_dataset()
    targets = np.array([y for _, y in dataset])
    model = Perceptron(
        size=2,
        low=-1.0,
        high=1.0,
        seed=42,
        use_bias=True,
        activation=sigmoid,
    )

    weights_loss = partial(
        dataset_loss,
        dataset,
        targets,
        bias=model.bias,
        activation=model.activation,
        loss=binary_cross_entropy,
    )
    numerical_weights = numerical_gradient(weights_loss, model.weights)

    bias_loss = partial(
        dataset_loss_given_bias,
        dataset,
        targets,
        model.weights,
        model.activation,
        loss=binary_cross_entropy,
    )
    numerical_bias = float(
        numerical_gradient(bias_loss, np.array([model.bias]))[0]
    )

    analytical_weights, analytical_bias = analytical_gradient(
        dataset,
        model.weights,
        bias=model.bias,
        activation=model.activation,
        loss="bce",
    )

    print(
        f"AND gate (sigmoid) - weights: {model.weights}, "
        f"bias: {model.bias:.6f}"
    )
    print(f"Numerical weight gradient:  {numerical_weights}")
    print(f"Analytical weight gradient: {analytical_weights}")
    print(f"Weight gradient difference: {np.abs(numerical_weights - analytical_weights)}")
    print(f"Numerical bias gradient:  {numerical_bias}")
    print(f"Analytical bias gradient: {analytical_bias}")
    print(f"Bias gradient difference: {abs(numerical_bias - analytical_bias)}")