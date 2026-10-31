from dl_lib.engine.value import Value


def main() -> None:
    a = Value(data=2, label="a")
    b = Value(data=-3, label="b")
    c = Value(data=10, label="c")
    
    e = a * b
    d = e + c

    f = Value(data=5)
    output = f * d

    output.grad = 1.0

    f.grad = output.grad * d.data
    d.grad = output.grad * f.data
    
    e.grad = d.grad
    c.grad = d.grad

    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    learning_rate = 0.01

    a.data -= learning_rate * a.grad
    b.data -= learning_rate * b.grad
    c.data -= learning_rate * c.grad
    f.data -= learning_rate * f.grad

    new_output = f * (a * b + c)

    print("L before step: ", output)
    print("L after step: ", new_output)
