from dl_lib.activations.functional import sigmoid
from dl_lib.losses.functional import binary_cross_entropy
from dl_lib.perceptron.datasets import nand_dataset
from dl_lib.perceptron.perceptron import Perceptron
from dl_lib.perceptron.training import train


def main() -> None:
    dataset = nand_dataset()
    model = Perceptron(
        size=2,
        low=-1.0,
        high=1.0,
        seed=42,
        use_bias=True,
        activation=sigmoid,
    )

    history = train(
        model,
        dataset,
        learning_rate=0.1,
        epochs=100_000,
        log_every=10_000,
        loss=binary_cross_entropy,
    )

    for epoch, loss in history:
        print(f"epoch {epoch}: loss={loss:.6f}")

    print(
        f"NAND gate - final weights: {model.weights}, "
        f"bias: {model.bias:.6f}"
    )
    print("NAND gate - predictions:")
    for x, target in dataset:
        prediction = model.forward(x)
        print(
            f"  NAND({x[0]:.0f}, {x[1]:.0f}) = {target:.0f} "
            f"-> predicted {prediction:.6f}"
        )
