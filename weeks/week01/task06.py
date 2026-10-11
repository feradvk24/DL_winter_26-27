import numpy as np


def print_one_d_array() -> None:
    one_d_array = np.array([1, 2, 3, 4, 5])
    print(
        f"1D array (vector): shape={one_d_array.shape}, "
        f"dtype={one_d_array.dtype}, ndim={one_d_array.ndim}, "
        f"size={one_d_array.size}"
    )
    print(one_d_array)
