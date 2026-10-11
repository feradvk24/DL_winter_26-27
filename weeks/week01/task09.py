import numpy as np

def add_scalar_to_matrix(matrix: np.ndarray, scalar: float) -> np.ndarray:
    return matrix + scalar

def main() -> None:
    matrix = np.arange(1, 7).reshape(2, 3)
    scalar = 5

    print(f"Matrix + scalar ({scalar}):")
    print(add_scalar_to_matrix(matrix, scalar))