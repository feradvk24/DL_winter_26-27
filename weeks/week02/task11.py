import matplotlib.pyplot as plt
import numpy as np
import torch

from dl_lib.activations.functional import relu, sigmoid, tanh


def main() -> None:
    x = np.linspace(-10, 10, 201)
    idx_minus_five, idx_zero, idx_five = np.searchsorted(x, [-5, 0, 5])

    sigmoid_values = sigmoid(x)
    relu_values = relu(x)
    tanh_values = tanh(x)

    print(f"sigmoid range: [{sigmoid_values.min():.4f}, {sigmoid_values.max():.4f}]")
    print(f"relu range:[{relu_values.min():.4f}, {relu_values.max():.4f}]")
    print(f"tanh range: [{tanh_values.min():.4f}, {tanh_values.max():.4f}]")
    print(f"At x=-5: sigmoid={sigmoid_values[idx_minus_five]:.4f}, relu={relu_values[idx_minus_five]:.4f}, tanh={tanh_values[idx_minus_five]:.4f}")
    print(f"At x=0: sigmoid={sigmoid_values[idx_zero]:.4f}, relu={relu_values[idx_zero]:.4f}, tanh={tanh_values[idx_zero]:.4f}")
    print(f"At x=5: sigmoid={sigmoid_values[idx_five]:.4f}, relu={relu_values[idx_five]:.4f}, tanh={tanh_values[idx_five]:.4f}")

    x_torch = torch.from_numpy(x)
    sigmoid_torch = torch.sigmoid(x_torch)
    relu_torch = torch.relu(x_torch)
    tanh_torch = torch.tanh(x_torch)

    print(f"PyTorch at x=-5: sigmoid={sigmoid_torch[idx_minus_five]:.4f}, relu={relu_torch[idx_minus_five]:.4f}, tanh={tanh_torch[idx_minus_five]:.4f}")
    print(f"PyTorch at x=0: sigmoid={sigmoid_torch[idx_zero]:.4f}, relu={relu_torch[idx_zero]:.4f}, tanh={tanh_torch[idx_zero]:.4f}")
    print(f"PyTorch at x=5: sigmoid={sigmoid_torch[idx_five]:.4f}, relu={relu_torch[idx_five]:.4f}, tanh={tanh_torch[idx_five]:.4f}")

    plt.plot(x, sigmoid_values, label="Sigmoid")
    plt.plot(x, relu_values, label="ReLU")
    plt.plot(x, tanh_values, label="Tanh")

    plt.xlabel("x")
    plt.ylabel("Activation output")
    plt.title("Sigmoid, ReLU, and tanh")
    plt.legend()
    plt.grid(True)
    plt.show()
