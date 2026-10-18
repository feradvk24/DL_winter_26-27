import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics.functional import apply_threshold, precision, recall


def print_metrics(idx: int, precisions: list[float], recalls: list[float], thresholds: np.ndarray):
    print(f"threshold={thresholds[idx]} precision={precisions[idx]}, recall={recalls[idx]}")

def main() -> None:
    rng = np.random.default_rng(42)

    y_true = rng.integers(0, 2, size=100)
    probabilities = np.empty(100)
    probabilities[y_true == 0] = rng.uniform(0.1, 0.6, size=(y_true == 0).sum())
    probabilities[y_true == 1] = rng.uniform(0.4, 0.9, size=(y_true == 1).sum())

    threshold_sweep = np.linspace(0, 1, 20)
    precisions = []
    recalls = []

    for thresh in threshold_sweep:
        classified = apply_threshold(probabilities, threshold=thresh)
        precisions.append(precision(y_true, classified))
        recalls.append(recall(y_true, classified))

    print_metrics(2, precisions, recalls, threshold_sweep)
    print_metrics(5, precisions, recalls, threshold_sweep)
    print_metrics(10, precisions, recalls, threshold_sweep)
    print_metrics(12, precisions, recalls, threshold_sweep)
    print_metrics(18, precisions, recalls, threshold_sweep)

    plt.plot(threshold_sweep, precisions, label="Precision", marker="o")
    plt.plot(threshold_sweep, recalls, label="Recall", marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and recall vs. threshold")
    plt.legend()
    plt.grid(True)
    plt.show()
