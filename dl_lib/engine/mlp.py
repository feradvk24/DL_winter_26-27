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


class MLP:
    def __init__(self, n_inputs: int, n_outputs: list[int], rng: random.Random | None = None):
        layer_rng = rng if rng else random.Random(42)

        self.layers = []
        current_inputs = n_inputs

        for output_size in n_outputs:
            layer = Layer(
                n_inputs = current_inputs,
                n_outputs = output_size,
                rng = layer_rng
            )
            self.layers.append(layer)
            current_inputs = output_size

    def __call__(self, inputs: list[Value]) ->  Value | list[Value]:
        outputs = inputs

        for layer in self.layers:
            outputs = layer(outputs)

        if len(outputs) == 1:
            return outputs[0]

        return outputs

    def parameters(self):
        return [
            parameter
            for layer in self.layers
            for parameter in layer.parameters()
        ]