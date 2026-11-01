import random

from dl_lib.engine.value import Value


class Neuron:
    def __init__(self, n_inputs: int, rng: random.Random | None = None):
        self.w = [Value(rng.uniform(-1, 1) if rng else random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(rng.uniform(-1, 1) if rng else random.uniform(-1, 1))

        
