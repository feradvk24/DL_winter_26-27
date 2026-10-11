import numpy as np
import torch


def print_np_array_info(arr: np.ndarray):
    print("NumPy array:")
    print(arr)
    print(f"dtype={arr.dtype}, shape={arr.shape}")


def main() -> None:
    arr = np.array([[1,2,3],[4,5,6]], dtype=float)
    print_np_array_info(arr)