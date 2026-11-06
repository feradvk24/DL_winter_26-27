import random

from dl_lib.engine.mlp import MLP


def main() -> int:
    mlp = MLP(3, [4, 4, 1], rng=random.Random(42))

    parameters = mlp.parameters()

    print("Number of parameters:", len(parameters))
    print("First three parameters:", [p.data for p in parameters[:3]])
    return 0