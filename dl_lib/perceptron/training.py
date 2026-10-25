from collections.abc import Callable
from functools import partial

import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.functional import (
    dataset_loss,
    dataset_loss_given_bias,
    gradient_descent_step,
    numerical_gradient,
)
from dl_lib.perceptron.perceptron import Perceptron


def train(
    model: Perceptron,
    dataset: list[tuple],
    learning_rate: float,
    epochs: int,
    log_every: int = 10,
    loss: Callable = mse,
    ) -> list[tuple[int, float]]: 
    targets = np.array([y for _, y in dataset])
    history = []

    for epoch in range(1, epochs + 1):
        weights_loss = partial(
            dataset_loss,
            dataset,
            targets,
            bias=model.bias,
            activation=model.activation,
            loss=loss,
        )

        gradient = numerical_gradient(weights_loss, model.weights)

        if model.use_bias:
            bias_loss = partial(
                dataset_loss_given_bias,
                dataset,
                targets,
                model.weights,
                model.activation,
                loss=loss,
            )
            bias_array = np.array([model.bias])
            bias_gradient = numerical_gradient(bias_loss, bias_array)

            model.bias = float(
                gradient_descent_step(
                    bias_array, bias_gradient, learning_rate
                )[0]
            )


        model.weights = gradient_descent_step(
            model.weights, gradient, learning_rate
        )

        if epoch % log_every == 0 or epoch == epochs:
            current_loss = dataset_loss(
                dataset,
                targets,
                model.weights,
                model.bias,
                model.activation,
                loss=loss,
            )
            history.append((epoch, current_loss))

    return history

        
