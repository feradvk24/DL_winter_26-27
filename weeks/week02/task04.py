import matplotlib.pyplot as plt
import numpy as np

from dl_lib.losses.functional import mae, mse, rmse


def main():
    y_true = np.array([3, 5, 7, 9])
    y_pred = np.array([3, 6, 8, 7])

    errors = np.arange(0, 11)
    mae_values = np.zeros(len(errors))
    mse_values = np.zeros(len(errors))
    rmse_values = np.zeros(len(errors))

    for i, error in enumerate(errors):
        y_pred[0] = y_true[0] + error
        mae_values[i] = mae(y_true, y_pred)
        mse_values[i] = mse(y_true, y_pred)
        rmse_values[i] = rmse(y_true, y_pred)

        print(
            f"Growing error={error:.1f}: MAE={mae_values[i]:.4f}, "
            f"MSE={mse_values[i]:.4f}, RMSE={rmse_values[i]:.4f}"
        )

    plt.figure()
    plt.plot(errors, mae_values, marker="o", label="MAE")
    plt.plot(errors, mse_values, marker="o", label="MSE")
    plt.plot(errors, rmse_values, marker="o", label="RMSE")
    plt.xlabel("The growing error")
    plt.ylabel("Loss value")
    plt.title("Loss metrics as one prediction error increases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
