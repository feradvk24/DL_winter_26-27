import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import forward as neuron_forward


class SquareNet:
    def __init__(
        self,
        hidden_units: int = 4,
        low: float = -1.0,
        high: float = 1.0,
        seed: int | None = None
    ):
        rng = np.random.default_rng(seed=seed)
        self.hidden_weights = rng.uniform(low, high, size=hidden_units)
        self.hidden_bias = rng.uniform(low, high, size=hidden_units)
        self.output_weights = rng.uniform(low, high, size=hidden_units)
        self.output_bias = float(rng.uniform(low, high))

    def forward(self, x: np.ndarray) -> float:
        input_value = x[0]
        hidden_logits = self.hidden_weights * input_value + self.hidden_bias
        hidden_outputs = sigmoid(hidden_logits)

        return neuron_forward(
            hidden_outputs,
            self.output_weights,
            self.output_bias,
        )