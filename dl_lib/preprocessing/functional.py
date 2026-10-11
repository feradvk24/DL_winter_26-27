import numpy as np


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return np.zeros_like(values, dtype=float)

    return (values - minimum) / (maximum - minimum)

def standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean()
    std = values.std()

    if std == 0:
        return np.zeros_like(values, dtype=float)

    return (values - mean) / std