import numpy as np


def create_plot(x, sin_values, cos_values, samples):
    pass

def main():
    samples = 1000
    x = np.linspace(0, 2 * np.pi, samples)
    sin_values = np.sin(x)
    cos_values = np.cos(x)
    samples = np.random.normal(size=samples)

    print(f"sin(x) range: [{np.min(sin_values):.4f}, {np.max(sin_values):.4f}]")
    print(f"cos(x) range: [{np.min(cos_values):.4f}, {np.max(cos_values):.4f}]")
    print(f"Histogram sample count: {len(samples)}")