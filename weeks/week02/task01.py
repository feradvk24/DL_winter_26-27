import numpy as np


def main():
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([2, 6, 8, 7])

    raw_errors = y_pred - y_true
    abs_errors = np.abs(raw_errors)
    squared_errors = np.power(raw_errors, 2)

    me = np.sum(raw_errors) / len(raw_errors)
    mae = np.sum(abs_errors) / len(abs_errors)
    mse = np.sum(squared_errors) / len(squared_errors)
    rmse = np.sqrt(mse)

    print("y_true: ", y_true)
    print("y_pred: ", y_pred)
    print("errors: ", raw_errors)
    print("absolute errors: ", abs_errors)
    print("squared errors: ", squared_errors)
    print(f"mean error: {me:.4f}")
    print(f"MAE (mean absolute error): {mae:.4f}")
    print(f"MSE (mean squared error):  {mse:.4f}")
    print(f"RMSE (root mean squared error): {rmse:.4f}")
