import numpy as np

def for_loop_sum(arr: np.ndarray) -> float:
    total = 0
    for num in arr:
        total += num
    return total

def compute_two_sums(arr: np.ndarray) -> tuple[float, float]:
    loop_sum = for_loop_sum(arr)
    vectorized_sum = np.sum(arr)
    return loop_sum, vectorized_sum

def main() -> None:
    large_arr = np.random.rand(1000000)
    loop_sum, vectorized_sum = compute_two_sums(large_arr)

    print(f"Array length: {len(large_arr):,}")
    print(f"Loop sum: {loop_sum}")
    print(f"Vectorized sum: {vectorized_sum}")