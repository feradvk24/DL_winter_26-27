import numpy as np

from dl_lib.metrics.functional import confusion_matrix


def main():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"True positives:  {conf_matrix['tp']}")
    print(f"True negatives:  {conf_matrix['tn']}")
    print(f"False positives: {conf_matrix['fp']}")
    print(f"False negatives: {conf_matrix['fn']}")


if __name__ == "__main__":
    main()
