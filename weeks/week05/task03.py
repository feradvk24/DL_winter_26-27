import random

from dl_lib.engine.mlp import Layer
from dl_lib.engine.value import Value


def main() -> int:
    layer = Layer(n_inputs=3, n_outputs=4, rng=random.Random(42))
    x = [Value(2.0), Value(3.0), Value(-1.0)]
    outputs = layer(x)

    print("layer(x) =", [output.data for output in outputs])
    print("Number of neurons:", len(layer.neurons))

    return 0
