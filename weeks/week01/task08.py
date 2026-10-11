import numpy as np

def for_loop_sum(arr: np.ndarray) -> float:
    total = 0
    for num in arr:
        total += num
    return total

def main() -> None:
    large_arr = np.random.rand(1000000)
    loop_sum = for_loop_sum(large_arr)
    print(f"Array length: {len(large_arr):,}")
    print(f"Loop sum: {loop_sum}")