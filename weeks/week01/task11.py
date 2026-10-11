import numpy as np

def create_dataset_with_noise(
    num_points: int,
    range_x: tuple[float, float],
    noise_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(range_x[0], range_x[1], num_points)
    noise = np.random.uniform(noise_range[0], noise_range[1], num_points)
    y = 2.0 * x + 1.0 + noise
    return x, y

def main() -> None:
    num_points = 50
    range_x = (0, 10)
    noise_range = (-1, 1)

    x, y = create_dataset_with_noise(num_points, range_x, noise_range)

    print(f"Generated {num_points} points for y = 2.0*x + 1.0 + noise")