import numpy as np


def apply_threshold(probability_array: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(probability_array < threshold, 0, 1)

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metrics = confusion_matrix(y_true, y_pred)

    return (metrics["tp"] + metrics["tn"]) / sum(list(metrics.values()))