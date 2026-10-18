import numpy as np

from dl_lib.metrics.functional import accuracy, f1_score, precision, recall


def main():
    y_true = np.array([0] * 990 + [1] * 10)
    y_pred = np.zeros_like(y_true)

    print("Dataset: 990 negative, 10 positive")
    print("Model: always predicts negative, never looks at the input")
    print(f"accuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"precision: {precision(y_true, y_pred):.4f}")
    print(f"recall: {recall(y_true, y_pred):.4f}")
    print(f"f1: {f1_score(y_true, y_pred):.4f}")
