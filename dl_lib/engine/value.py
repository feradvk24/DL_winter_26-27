import math


class Value:
    def __init__(
        self, 
        data: float,
        _prev: set | None=None, 
        _op: str | None=None, 
        label="",
        grad: float=0.0,
        ):
        self.data = data
        self._prev = _prev if _prev else set()
        self._op = _op
        self.label = label
        self.grad = grad


    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Value(data=(self.data + other), _prev={self}, _op="+")
        elif isinstance(other, Value):
            return Value(data=(self.data + other.data), _prev={self, other}, _op="+")
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Value(data=(self.data * other), _prev={self}, _op="*")
        elif isinstance(other, Value):
            return Value(data=(self.data * other.data), _prev={self, other}, _op="*")
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

    def tanh(self) -> "Value":
        return Value(data=math.tanh(self.data), _prev={self}, _op="tanh")

