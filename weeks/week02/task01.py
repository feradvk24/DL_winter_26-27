import numpy as np


def main():
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    abs_errors = np.abs(y_pred - y_true)
    squared_errors = np.power(y_pred - y_true, 2)
    print(y_true)
    print(y_pred)
    print("absolute errors: ", abs_errors)
    print("squared_errors: ", squared_errors)