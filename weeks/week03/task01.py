from dl_lib.perceptron.datasets import create_dataset
from dl_lib.perceptron.functional import initialize_weights


def main() -> None:
    data = [(float(x), float(y)) for x, y in create_dataset(4)]
    weight_100 = initialize_weights(0, 100, size=1, seed=42)[0]
    weight_10 = initialize_weights(0, 10, size=1, seed=42)[0]

    print(f"Dataset: {data}")
    print(f"initialize_weights(0, 100): {weight_100}")
    print(f"initialize_weights(0, 10):  {weight_10}")
