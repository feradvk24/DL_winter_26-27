import time

import numpy as np

def for_loop_sum(arr: np.ndarray) -> float:
    total = 0
    for num in arr:
        total += num
    return total

def compute_two_sums(arr: np.ndarray) -> tuple[float, float, float, float]:
    # Measure loop sum and time
    start_time = time.perf_counter()
    loop_sum = for_loop_sum(arr)
    loop_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    vectorized_sum = np.sum(arr)
    vectorized_time = time.perf_counter() - start_time

    return loop_sum, vectorized_sum, loop_time, vectorized_time

def get_array_stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    mean = np.mean(arr)
    std_dev = np.std(arr)
    min_val = np.min(arr)
    max_val = np.max(arr)

    return mean, std_dev, min_val, max_val

def main() -> None:
    large_arr = np.random.rand(1000000)
    loop_sum, vectorized_sum, loop_time, vectorized_time = compute_two_sums(large_arr)
    mean, std_dev, min_val, max_val = get_array_stats(large_arr)
    
    print(f"Array length: {len(large_arr):,}")
    print(f"Loop sum: {loop_sum:.4f} ({loop_time:.4f}s)")
    print(
        f"Vectorized sum: {vectorized_sum:.4f} "
        f"({vectorized_time:.4f}s)"
    )
    print(f"Speedup: {loop_time / vectorized_time:.2f}x")
    print(f"Vectorized mean: {mean:.4f}")
    print(f"Vectorized std: {std_dev:.4f}")
    print(f"Vectorized min/max: {min_val:.4f} / {max_val:.4f}")