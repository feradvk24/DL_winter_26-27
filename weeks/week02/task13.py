import numpy as np

from dl_lib.losses.functional import binary_cross_entropy


def main() -> None:
    y_true = np.array([1.0, 0.0, 1.0])
    probabilities = np.array([0.9, 0.2, 0.6])
    loss = binary_cross_entropy(y_true, probabilities)
    print(f"Binary cross-entropy on y_true={y_true}, p={probabilities}: {loss:.4f}")

    y_true = np.array([1])
    right_loss = binary_cross_entropy(y_true, np.array([0.999]))
    wrong_loss = binary_cross_entropy(y_true, np.array([0.001]))

    print()
    print(f"y_true=1, p=0.999 (confidently right): loss={right_loss:.4f}")
    print(f"y_true=1, p=0.001 (confidently wrong): loss={wrong_loss:.4f}")