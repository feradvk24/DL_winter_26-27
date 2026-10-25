from dl_lib.perceptron.datasets import and_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train
from dl_lib.activations.functional import sigmoid


def main() -> None:
    dataset = and_dataset()

    model = Perceptron(
        size=2,
        low=-1.0,
        high=1.0,
        seed=42,
        use_bias=True,
        activation=None,
    )

    history = train(
        model,
        dataset,
        learning_rate=0.1,
        epochs=100_000,
        log_every=10_000,
    )

    model_with_activation = Perceptron(
        size=2,
        low=-1.0,
        high=1.0,
        seed=42,
        use_bias=True,
        activation=sigmoid,
    )

    history_with_activation = train(
        model_with_activation,
        dataset,
        learning_rate=0.1,
        epochs=100_000,
        log_every=10_000,
    )

    print(f"Without sigmoid - final loss: {history[-1][1]}")
    for x, target in dataset:
        prediction = model.forward(x)
        print(
            f"AND({x[0]:.0f}, {x[1]:.0f}) = {target:.0f} "
            f"-> predicted {prediction:.6f}"
        )
    print(f"With sigmoid    - final loss: {history_with_activation[-1][1]}")
    for x, target in dataset:
        prediction = model_with_activation.forward(x)
        print(
            f"AND({x[0]:.0f}, {x[1]:.0f}) = {target:.0f} "
            f"-> predicted {prediction:.6f}"
        )
