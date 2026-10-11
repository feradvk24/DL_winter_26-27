import numpy as np


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros_like(values, dtype=float)

    return (values - minimum) / (maximum - minimum)