import numpy as np

def for_loop_sum(arr: np.ndarray) -> float:
    total = 0
    for num in arr:
        total += num
    return total