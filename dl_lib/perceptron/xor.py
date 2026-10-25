import numpy as np

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.functional import forward as neuron_forward


class Xor:
    def __init__(
        self,
        low: float = 0,
        high: float = 1,
        seed: int | None = None,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.hidden_weights = rng.uniform(low, high, size=(2, 2))
        self.hidden_bias = rng.uniform(low, high, size=2)
        self.output_weights = rng.uniform(low, high, size=2)
        self.output_bias = float(rng.uniform(low, high))

    def forward(self, x: np.ndarray) -> float:
        h1 = neuron_forward(
            x,
            self.hidden_weights[0],
            self.hidden_bias[0],
            activation=sigmoid,
        )

        h2 = neuron_forward(
            x,
            self.hidden_weights[1],
            self.hidden_bias[1],
            activation=sigmoid,
        )

        hidden_outputs = np.array([h1, h2])

        return neuron_forward(
            hidden_outputs,
            self.output_weights,
            self.output_bias,
            activation=sigmoid,
        )