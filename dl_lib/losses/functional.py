import numpy as np


def mae(y_true, y_pred):
    return np.sum(np.abs(y_pred - y_true)) / len(y_true)