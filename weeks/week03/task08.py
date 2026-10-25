import matplotlib.pyplot as plt

from dl_lib.activations.functional import sigmoid
from dl_lib.perceptron.datasets import and_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


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
        log_every=1,
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
        log_every=1,
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

    epochs, losses = zip(*history)
    sigmoid_epochs, sigmoid_losses = zip(*history_with_activation)

    plt.figure()
    plt.plot(epochs, losses, label="no activation")
    plt.plot(sigmoid_epochs, sigmoid_losses, label="sigmoid")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("AND gate - loss over 100,000 epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
