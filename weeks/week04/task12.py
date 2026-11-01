from dl_lib.engine.value import Value


def main() -> None:
    a = Value(3.0)
    b = a + a

    b.backward()

    print("b.data =", b.data)
    print("a.grad =", a.grad)