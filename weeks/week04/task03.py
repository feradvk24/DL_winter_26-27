from io import BytesIO

import matplotlib.pyplot as plt

from dl_lib.engine.graph import draw_dot, trace
from dl_lib.engine.value import Value


def main() -> None:
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    result = a * b + c

    nodes, edges = trace(result)
    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of edges: {len(edges)}")

    dot = draw_dot(result)
    image = plt.imread(BytesIO(dot.pipe(format="png")), format="png")
    figure, axis = plt.subplots(figsize=(16, 4))
    axis.imshow(image)
    axis.axis("off")
    figure.tight_layout()
    plt.show()
