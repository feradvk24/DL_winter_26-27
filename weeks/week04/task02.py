from dl_lib.engine.value import Value


def main() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)

    e = a * b
    d = e + c

    d_prev = list(d._prev)
    e_prev = list(e._prev)

    print("d =", d)
    print("d._op =", repr(d._op))
    print("d._prev =", sorted([d_prev[0].data, d_prev[1].data], reverse=True))
    print("e =", e)
    print("e._op =", repr(e._op))
    print("e._prev =", sorted([e_prev[0].data, e_prev[1].data], reverse=True))
