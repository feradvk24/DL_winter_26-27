import random

from dl_lib.engine.mlp import MLP
from dl_lib.engine.value import Value


def main() -> None:
    mlp = MLP(n_inputs=3, n_outputs=[4, 4, 1], rng=random.Random(42))
    x = [Value(2.0), Value(3.0), Value(-1.0)]

    print("mlp(x) =", mlp(x))
    print("Number of layers:", len(mlp.layers))
