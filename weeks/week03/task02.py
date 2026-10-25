import numpy as np

from dl_lib.losses.functional import mse
from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import forward, initialize_weights


def main() -> None:
    data = create_dataset(6)
    x = np.array([[sample[0]] for sample in data], dtype=float)
    y_true = np.array([sample[1] for sample in data], dtype=float)

    weights = initialize_weights(0, 10, size=1, seed=42)
    y_pred = np.array([forward(inputs, weights) for inputs in x])
    loss = mse(y_true, y_pred)

    print(f"Dataset: {data}")
    print(f"w: {weights[0]}")
    print(f"MSE: {loss}")
    print("\nPerturbing w:")

    for nudge in [0.002, 0.001, -0.001, -0.002]:
        new_weights = weights + nudge
        y_pred = np.array([forward(inputs, new_weights) for inputs in x])
        loss = mse(y_true, y_pred)

        print(f"  w + ({nudge:+.3f}) -> w={new_weights[0]:.6f}, MSE={loss:.6f}")
