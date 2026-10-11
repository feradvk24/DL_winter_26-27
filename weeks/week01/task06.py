import numpy as np


def print_one_d_array() -> None:
    one_d_array = np.array([1, 2, 3, 4, 5])
    print(
        f"1D array (vector): shape={one_d_array.shape}, "
        f"dtype={one_d_array.dtype}, ndim={one_d_array.ndim}, "
        f"size={one_d_array.size}"
    )
    print(one_d_array)

def print_two_d_array() -> None:
    two_d_array = np.arange(1, 7).reshape(2, 3)
    print(
        f"2D array (matrix): shape={two_d_array.shape}, "
        f"dtype={two_d_array.dtype}, ndim={two_d_array.ndim}, "
        f"size={two_d_array.size}"
    )
    print(two_d_array)

def print_three_d_array() -> None:
    three_d_array = np.arange(24).reshape(2, 3, 4)
    print(
        f"3D array (volume): shape={three_d_array.shape}, "
        f"dtype={three_d_array.dtype}, ndim={three_d_array.ndim}, "
        f"size={three_d_array.size}"
    )
    print(three_d_array)