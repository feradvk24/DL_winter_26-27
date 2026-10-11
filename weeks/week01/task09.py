import numpy as np

def add_scalar_to_matrix(matrix: np.ndarray, scalar: float) -> np.ndarray:
    return matrix + scalar

def add_vector_to_matrix(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return matrix + vector


def main() -> None:
    matrix = np.arange(1, 7).reshape(2, 3)
    scalar = 5
    incompatible_vector = np.array([10, 20, 30, 40])
    row_vector = np.array([10, 20, 30])
    column_vector = np.array([[10], [20]])

    print(f"Matrix + scalar ({scalar}):")
    print(add_scalar_to_matrix(matrix, scalar))

    print(f"Matrix {matrix.shape} + row vector {row_vector.shape}:")
    print(add_vector_to_matrix(matrix, row_vector))

    print(f"Matrix {matrix.shape} + column vector {column_vector.shape}:")
    print(add_vector_to_matrix(matrix, column_vector))

    try:
        print(f"Matrix {matrix.shape} + incompatible vector {incompatible_vector.shape}:")
        print(add_vector_to_matrix(matrix, incompatible_vector))
    except ValueError as e:
        print(f"operands could not be broadcast together with shapes: {matrix.shape} {incompatible_vector.shape}")