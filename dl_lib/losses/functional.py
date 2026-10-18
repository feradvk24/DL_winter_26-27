import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.abs(y_pred - y_true)) / len(y_true)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(np.power(y_pred - y_true, 2)) / len(y_true)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    epsilon = 1e-7
    p = np.clip(y_pred, epsilon, 1-epsilon)

    cross_entropy_losses = (y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return -np.mean(cross_entropy_losses)

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,) -> float:
    epsilon = 1e-7
    p = np.clip(y_pred, epsilon, 1 - epsilon)

    rows = np.arange(len(y_true))
    correct_probabilities = p[rows, y_true]

    losses = np.log(correct_probabilities)
    return -np.mean(losses)