import time

import numpy as np

def for_loop_sum(arr: np.ndarray) -> float:
    total = 0
    for num in arr:
        total += num
    return total

def compute_two_sums(arr: np.ndarray) -> tuple[float, float, float, float]:
    # Measure loop sum and time
    start_time = time.time()
    loop_sum = for_loop_sum(arr)
    loop_time = time.time() - start_time
    start_time = time.time()
    vectorized_sum = np.sum(arr)
    vectorized_time = time.time() - start_time

    return loop_sum, vectorized_sum, loop_time, vectorized_time

def main() -> None:
    large_arr = np.random.rand(1000000)
    loop_sum, vectorized_sum, loop_time, vectorized_time = compute_two_sums(large_arr)

    print(f"Array length: {len(large_arr):,}")
    print(f"Loop sum: {loop_sum}")
    print(f"Vectorized sum: {vectorized_sum}")