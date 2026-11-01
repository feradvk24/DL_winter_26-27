import random

from dl_lib.engine.value import Value


class Neuron:
    def __init__(self, n_inputs: int, rng: random.Random | None = None):
        self.w = [Value(rng.uniform(-1, 1) if rng else random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(rng.uniform(-1, 1) if rng else random.uniform(-1, 1))

    def __call__(self, inputs: list[Value]) -> Value:
        if len(inputs) != len(self.w):
            raise ValueError("Number of inputs must match number of weights.")
        
        result = sum(w_i.data * x_i.data for w_i, x_i in zip(self.w, inputs))
        return Value(data=result + self.b.data).tanh()

    def parameters(self) -> list[Value]:
        return self.w + [self.b]


class Layer:
    def __init__(self, n_inputs: int, n_outputs: int, rng: random.Random | None = None):
        neuron_rng = rng if rng else random.Random(42)
        self.neurons = [Neuron(n_inputs=n_inputs, rng=neuron_rng) for _ in range(n_outputs)]

    def __call__(self, inputs: list[Value]) -> list[Value]:
        return [neuron(inputs) for neuron in self.neurons]

    def parameters(self):
        return [
            parameter
            for neuron in self.neurons
            for parameter in neuron.parameters()
        ]