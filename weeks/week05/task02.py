import random

from dl_lib.engine.mlp import Neuron
from dl_lib.engine.value import Value


def main() -> int:
    neuron = Neuron(3, rng=random.Random(42))
    x = [Value(2.0), Value(3.0), Value(-1.0)]

    print("neuron.w =", [weight.data for weight in neuron.w])
    print("neuron.b =", neuron.b.data)
    print("neuron(x) =", neuron(x))

    return 0