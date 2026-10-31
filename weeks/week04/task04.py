from io import BytesIO

import matplotlib.pyplot as plt

from dl_lib.engine.graph import draw_dot, trace
from dl_lib.engine.value import Value


def main():
    a = Value(data=2, label="a")
    b = Value(data=-3, label="b")
    c = Value(data=10, label="c")
    
    e = a * b
    e.label = "e"

    d = e + c
    d.label= "d"

    f = Value(data=5, label="f")
    output = f * d
    output.label = "output"

    graph = draw_dot(output)

    print("output = ", output)
    print("output.grad = ", output.grad)
    print("d.grad = ", d.grad)

    image = plt.imread(BytesIO(graph.pipe(format="png")), format="png")

    figure, axis = plt.subplots(figsize=(16, 4))
    axis.imshow(image)
    axis.axis("off")
    figure.tight_layout()
    plt.show()
