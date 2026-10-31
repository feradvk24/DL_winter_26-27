from io import BytesIO

import matplotlib.pyplot as plt

from dl_lib.engine.graph import draw_dot
from dl_lib.engine.value import Value


def main() -> None:
    x_1 = Value(data=2.0, label="x1")
    x_2 = Value(data=0.0, label="x2")
    w_1 = Value(data=-3.0, label="w1")
    w_2 = Value(data=1.0, label="w2")
    b = Value(data=6.8813735870195432, label="b")

    x1w1 = x_1 * w_1
    x1w1.label = "x1*w1"

    x2w2 = x_2 * w_2
    x2w2.label = "x2*w2"

    weighted_sum = x1w1 + x2w2
    weighted_sum.label = "x1*w1 + x2*w2"

    logit = weighted_sum + b
    logit.label = "logit"
    print("logit =", logit)

    graph = draw_dot(logit)
    image = plt.imread(BytesIO(graph.pipe(format="png")), format="png")

    figure, axis = plt.subplots(figsize=(16, 4))
    axis.imshow(image)
    axis.axis("off")
    figure.tight_layout()
    plt.show()
