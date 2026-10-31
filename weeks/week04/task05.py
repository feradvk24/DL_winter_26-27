from collections.abc import Callable

from dl_lib.engine.value import Value


def manual_der(
    function: Callable[..., float],
    *arguments: float,
) -> tuple:
    epsilon = 1e-7

    derivatives = []
    for index in range(len(arguments)):
        plus = list(arguments)
        minus = list(arguments)
        plus[index] += epsilon
        minus[index] -= epsilon
        derivative = (function(*plus) - function(*minus)) / (2 * epsilon)
        derivatives.append(derivative)

    return tuple(derivatives)


def main() -> None:
    a = Value(data=2, label="a")
    b = Value(data=-3, label="b")
    c = Value(data=10, label="c")
    
    e = a * b
    e.label = "e"

    d = e + c
    d.label= "d"

    f = Value(data=5, label="f")
    output = f * d

    output.grad = 1.0

    f.grad = output.grad * d.data
    d.grad = output.grad * f.data
    
    e.grad = d.grad
    c.grad = d.grad

    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    numerical = manual_der(
        lambda a, b, c, f: f * (a * b + c),
        a.data, b.data, c.data, f.data,
    )
    values = (a, b, c, f)
    print("Manual gradients: " + ", ".join(
        f"{node.label}={node.grad}" for node in values
    ))
    print("Numerical gradients: " + ", ".join(
        f"{node.label}={gradient}" for node, gradient in zip(values, numerical)
    ))