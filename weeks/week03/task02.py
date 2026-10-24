import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights


def main() -> None:
    data = create_dataset(6)
    x = np.array([sample[0] for sample in data], dtype=float)
    y_true = np.array([sample[1] for sample in data], dtype=float)

    weight = initialize_weights(0, 10, size=1, seed=42)[0]
    y_pred = weight * x
    loss = mse(y_true, y_pred)

    for nudge in [0.002, 0.001, -0.001, -0.002]:
        new_weight = weight + nudge
        y_pred = new_weight * x
        loss = mse(y_true, y_pred)

        print(f"  w + ({nudge:+.3f}) -> w={new_weight:.6f}, MSE={loss:.6f}")
