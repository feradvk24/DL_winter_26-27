import numpy as np


def create_dataset(n: int) -> list[tuple[int, int]]:
    return [(i, 2*i) for i in range(n)]

def and_dataset() -> list[tuple[np.ndarray, float]]:
    return [
        (np.array([0.0, 0.0]), 0.0),
        (np.array([0.0, 1.0]), 0.0),
        (np.array([1.0, 0.0]), 0.0),
        (np.array([1.0, 1.0]), 1.0),
    ]

def or_dataset() -> list[tuple[np.ndarray, float]]:
    return [
        (np.array([0.0, 0.0]), 0),
        (np.array([0.0, 1.0]), 1.0),
        (np.array([1.0, 0.0]), 1.0),
        (np.array([1.0, 1.0]), 1.0),
    ]