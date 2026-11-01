from dl_lib.engine.value import Value


class _OverwritingValue(Value):
    def __add__(self, other: "_OverwritingValue") -> "_OverwritingValue":
        out = _OverwritingValue(
            self.data + other.data, _prev={self, other}, _op="+"
        )

        def _backward():
            self.grad = out.grad
            other.grad = out.grad

        out._backward = _backward
        return out


def main() -> None:
    a = _OverwritingValue(3.0)
    b = a + a

    b.backward()

    print("b.data =", b.data)
    print("a.grad =", a.grad)