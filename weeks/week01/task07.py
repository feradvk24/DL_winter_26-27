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

def print_boolean_mask_selection(matrix: np.ndarray) -> None:
    boolean_mask = matrix > 60
    selected_elements = matrix[boolean_mask]
    print(f"Boolean mask for more than 60:\n{boolean_mask}")
    print(f"Values satisfying the mask: {selected_elements}")

def main() -> None:
    matrix = build_matrix()
    print_row_and_column(matrix)
    print_matrix_slice(matrix)
    print_boolean_mask_selection(matrix)