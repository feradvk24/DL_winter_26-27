import numpy as np
import torch


def print_np_array_info(arr: np.ndarray):
    print("NumPy array:")
    print(arr)
    print(f"dtype={arr.dtype}, shape={arr.shape}")

def print_tensor_info(tensor: torch.Tensor) -> None:
    print("PyTorch tensor:")
    print(tensor)
    print(f"dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")


def main() -> None:
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    print_np_array_info(arr)
    tensor = torch.from_numpy(arr)
    print_tensor_info(tensor)
