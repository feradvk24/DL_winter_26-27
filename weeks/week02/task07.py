import numpy as np

from dl_lib.metrics.functional import accuracy, f1_score, precision, recall


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    print(f"  accuracy:  {accuracy(y_true, y_pred):.4f}")
    print(f"  precision: {precision(y_true, y_pred):.4f}")
    print(f"  recall:    {recall(y_true, y_pred):.4f}")
    print(f"  f1:        {f1_score(y_true, y_pred):.4f}")


def main():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    print("Balanced dataset:")
    print_metrics(y_true, y_pred)

    y_true = np.array([1] * 10 + [0] * 990)
    y_pred = y_true.copy()
    y_pred[:3] = 0

    print("Imbalanced dataset (990 negative, 10 positive):")
    print_metrics(y_true, y_pred)
