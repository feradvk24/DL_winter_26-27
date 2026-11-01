from dl_lib.engine.value import Value


def main() -> None:
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = a + b
    c.label = "c"

    c.backward()

    print("c =", c)
    print("a.grad =", a.grad)
    print("b.grad =", b.grad)