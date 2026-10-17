import numpy as np

from dl_lib.metrics.functional import apply_threshold


def main():
    probabilities_array = np.array([0.1, 0.8, 0.4, 0.7, 0.9])

    threshold_arr = np.linspace(start=0.1, stop=0.9, num=9).round(1)

    for thresh in threshold_arr:
        predictions = apply_threshold(probabilities_array, threshold=thresh)
        print(f"Threshold={thresh:.1f}: predictions={predictions}")
