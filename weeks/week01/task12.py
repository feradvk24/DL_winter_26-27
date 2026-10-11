import matplotlib.pyplot as plt
import numpy as np


def create_plot_data(samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0, 2 * np.pi, samples)
    sin_values = np.sin(x)
    cos_values = np.cos(x)
    hist_data = np.random.normal(size=samples)
    return x, sin_values, cos_values, hist_data

def create_plot(x, sin_values, cos_values, hist_data):
    figure, axes = plt.subplots(1, 2, figsize=(10, 8))
    line_axis, histogram_axis = axes

    line_axis.plot(x, sin_values, label="sin(x)", color="blue")
    line_axis.plot(x, cos_values, label="cos(x)", color="orange")
    line_axis.legend()
    line_axis.set_title("Line Plot")
    line_axis.set_xlabel("x")
    line_axis.set_ylabel("y")
    line_axis.grid(True)

    histogram_axis.hist(hist_data)
    histogram_axis.set_title("Histogram")
    histogram_axis.set_xlabel("value")
    histogram_axis.set_ylabel("count")
    histogram_axis.grid(True)

    figure.tight_layout()

    return figure, axes

def main():
    samples = 1000
    x, sin_values, cos_values, hist_data = create_plot_data(samples)

    print(f"sin(x) range: [{np.min(sin_values):.4f}, {np.max(sin_values):.4f}]")
    print(f"cos(x) range: [{np.min(cos_values):.4f}, {np.max(cos_values):.4f}]")
    print(f"Histogram sample count: {len(hist_data)}")

    create_plot(x, sin_values, cos_values, hist_data)
    plt.show()