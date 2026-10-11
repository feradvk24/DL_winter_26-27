import numpy as np

def build_matrix() -> np.ndarray:
    return np.arange(1, 13).reshape(3, 4)

def print_row_and_column(matrix: np.ndarray) -> None:
    second_row = matrix[1, :]
    third_column = matrix[:, 2]
    print(f"Second row: {second_row}")
    print(f"Third column: {third_column}")

def print_matrix_slice(matrix: np.ndarray) -> None:
    matrix_slice = matrix[0:2, 1:3]
    print(f"Submatrix:\n{matrix_slice}")