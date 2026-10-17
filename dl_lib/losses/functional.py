import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_pred - y_true)) / len(y_true)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.power(y_pred - y_true, 2)) / len(y_true)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))