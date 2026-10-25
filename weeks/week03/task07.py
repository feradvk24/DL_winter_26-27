from dl_lib.perceptron.datasets import and_dataset, or_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


def main() -> None:
    for name, dataset in (("AND", and_dataset()), ("OR", or_dataset())):
        model = Perceptron(
            size=2,
            low=-1.0,
            high=1.0,
            seed=42,
            use_bias=True,
            activation=None,
        )
        print(
            f"{name} gate - initial weights: {model.weights}, "
            f"bias: {model.bias:.6f}"
        )

        history = train(
            model,
            dataset,
            learning_rate=0.1,
            epochs=100_000,
            log_every=10_000,
        )

        for epoch, loss in history:
            print(f"epoch {epoch}: loss={loss:.6f}")

        print(
            f"{name} gate - final weights: {model.weights}, "
            f"bias: {model.bias:.6f}"
        )
        print(f"{name} gate - predictions:")
        for x, target in dataset:
            prediction = model.forward(x)
            print(
                f"  {name}({x[0]:.0f}, {x[1]:.0f}) = {target:.0f} "
                f"-> predicted {prediction:.6f}"
            )
        print()
