import numpy as np

def main() -> None:
    matrix_a = np.array([[1, 2], [3, 4]])
    matrix_b = np.array([[5, 6], [7, 8]])
    elementwise_product = matrix_a * matrix_b
    matrix_product = matrix_a @ matrix_b

    print(f"A:\n{matrix_a}")
    print(f"B:\n{matrix_b}")
    print(f"Elementwise:\n{elementwise_product}")
    print(f"Matrix multiplication:\n{matrix_product}")
    print(f"Transpose of A:\n{matrix_a.T}")