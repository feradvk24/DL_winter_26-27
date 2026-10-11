import numpy as np

def add_scalar_to_matrix(matrix: np.ndarray, scalar: float) -> np.ndarray:
    return matrix + scalar

def add_vector_to_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return matrix + vector


def main() -> None:
    matrix = np.arange(1, 7).reshape(2, 3)
    scalar = 5

    print(f"Matrix + scalar ({scalar}):")
    print(add_scalar_to_matrix(matrix, scalar))
    print(f"Matrix {matrix.shape} + row vector {np.array([10, 20, 30]).shape}:")
    print(add_vector_to_matrix(matrix, np.array([10, 20, 30])))
    print(f"Matrix {matrix.shape} + column vector {np.array([[10], [20]]).shape}:")
    print(add_vector_to_matrix(matrix, np.array([[10], [20]])))