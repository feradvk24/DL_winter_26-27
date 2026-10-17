import numpy as np

from dl_lib.losses.functional import mse

def main():
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    print("MSE: ", mse(y_true, y_pred))
