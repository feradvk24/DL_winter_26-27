from dl_lib.engine.value import Value


def main():
    a = Value(data=2.0)
    b = Value(data=-3.0)
    c = Value(10.0)

    print(a)
    print(b)
    print(c)

    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a * b + c = {a * b + c}")