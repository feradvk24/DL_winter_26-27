import numpy as np
import torch

from dl_lib.activations.functional import softmax


def naive_softmax(x: np.ndarray) -> np.ndarray:
    exp_values = np.exp(x)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def main() -> None:
    scores = np.array([1, 2, 3], dtype=float)
    softmax_scores = softmax(scores)

    print("Softmax: ", softmax_scores)
    print(f"Sum of probabilities: {np.sum(softmax_scores):4g}")

    large_scores = np.array([1000.0, 1001.0, 1002.0])
    naive_softmax_scores = naive_softmax(large_scores)
    reliable_softmax_scores = softmax(large_scores)

    print("Large scores:", large_scores)
    print("Naive softmax (unstable):", naive_softmax_scores)
    print("Stable softmax:", reliable_softmax_scores)

    torch_scores = torch.softmax(torch.from_numpy(scores), dim=-1)
    torch_large_scores = torch.softmax(torch.from_numpy(large_scores), dim=-1)
    print("PyTorch softmax (small scores):", torch_scores.numpy())
    print("PyTorch softmax (large scores):", torch_large_scores.numpy())
