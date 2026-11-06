import numpy as np

from dl_lib.engine.mlp import MLP
from dl_lib.engine.value import Value
from dl_lib.losses.functional import mse


def main() -> int:
    mlp = MLP(3, [4, 4, 1])

    samples = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    targets = [1.0, -1.0, -1.0, 1.0]

    predictions = [
        mlp([Value(number) for number in sample])
        for sample in samples
    ]
    predictions_list = [value.data for value in predictions]

    error = mse(np.array(targets), np.array(predictions_list))

    print("Predictions: ", predictions)
    print("MSE loss: ", error * len(predictions_list))

    return 0