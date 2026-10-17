import numpy as np


def apply_threshold(probability_array: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(probability_array < threshold, 0, 1)
