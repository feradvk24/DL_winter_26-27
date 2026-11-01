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
        self._backward = lambda: None


    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def backward(self) -> None:
        from dl_lib.engine.graph import top_sort

        self.grad = 1.0
        for node in reversed(top_sort(self)):
            node._backward()

    def __add__(self, other):
        if isinstance(other, (int, float)):
            out = Value(data=(self.data + other), _prev={self}, _op="+")

            def _backward():
                self.grad += out.grad

        elif isinstance(other, Value):
            out = Value(data=(self.data + other.data), _prev={self, other}, _op="+")

            def _backward():
                self.grad += out.grad
                other.grad += out.grad

        else:
            return NotImplemented

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            out = Value(data=(self.data * other), _prev={self}, _op="*")

            def _backward():
                self.grad += other * out.grad

        elif isinstance(other, Value):
            out = Value(data=(self.data * other.data), _prev={self, other}, _op="*")

            def _backward():
                self.grad += other.data * out.grad
                other.grad += self.data * out.grad

        else:
            return NotImplemented

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def tanh(self) -> "Value":
        out = Value(data=math.tanh(self.data), _prev={self}, _op="tanh")

        def _backward():
            self.grad += (1.0 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> "Value":
        out = Value(data=self.data**exponent, _prev={self}, _op=f"**{exponent}")

        def _backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out