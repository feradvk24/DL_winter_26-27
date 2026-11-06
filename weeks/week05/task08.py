from dl_lib.engine.mlp import MLP
from dl_lib.engine.value import Value


def predict(mlp: MLP, inputs: list[Value]) -> Value:
    """Build a differentiable forward pass using the existing parameters."""
    outputs = inputs
    for layer in mlp.layers:
        next_outputs = []
        for neuron in layer.neurons:
            weighted_sum = sum(
                weight * value for weight, value in zip(neuron.w, outputs)
            )
            next_outputs.append((weighted_sum + neuron.b).tanh())
        outputs = next_outputs

    return outputs[0]


def main() -> int:
    mlp = MLP(3, [4, 4, 1])

    samples = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    targets = [1.0, -1.0, -1.0, 1.0]

    parameters = mlp.parameters()
    learning_rate = 0.05

    for epoch in range(20):
        predictions = [
            predict(mlp, [Value(number) for number in sample])
            for sample in samples
        ]

        loss = sum(
            (prediction + (-target)) ** 2
            for prediction, target in zip(predictions, targets)
        )

        for parameter in parameters:
            parameter.grad = 0.0

        loss.backward()

        total_gradient = sum(abs(p.grad) for p in parameters)
        print(
            f"epoch {epoch}: loss = {loss.data}, "
            f"sum|grad| = {total_gradient}"
        )

        for parameter in parameters:
            parameter.data -= learning_rate * parameter.grad


    return 0
