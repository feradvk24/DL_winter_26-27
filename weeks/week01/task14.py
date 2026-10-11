import numpy as np

from dl_lib.preprocessing.functional import min_max_normalize, standardize


def main() -> None:
    values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    normalized = min_max_normalize(values)
    standardized = standardize(values)

    print(f"Original:     {values}")
    print(f"Min-max:      {normalized}")
    print(f"Standardized: {standardized}")
    print()
    print(f"Min-max range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(
        f"Standardized mean/std: {standardized.mean():.4f} / "
        f"{standardized.std():.4f}"
    )
