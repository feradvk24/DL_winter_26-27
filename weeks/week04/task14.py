from dl_lib.engine.value import Value


def get_logit(x_1, x_2, w_1, w_2, b) -> Value:
    x1w1 = x_1 * w_1
    x1w1.label = "x1*w1"

    x2w2 = x_2 * w_2
    x2w2.label = "x2*w2"

    weighted_sum = x1w1 + x2w2
    weighted_sum.label = "x1*w1 + x2*w2"

    logit = weighted_sum + b
    logit.label = "logit"

    return logit


def main() -> None:
    x_1 = Value(data=2.0, label="x1")
    x_2 = Value(data=0.0, label="x2")
    w_1 = Value(data=-3.0, label="w1")
    w_2 = Value(data=1.0, label="w2")
    b = Value(data=6.8813735870195432, label="b")

    logit = get_logit(x_1, x_2, w_1, w_2, b)
    print("logit =", logit)

    doubled = 2 * logit
    exponential = doubled.exp()

    numerator = exponential + -1.0
    denominator = exponential + 1.0

    output = numerator / denominator

    output.backward()

    print("o decomposed: ", output)
    print("x1.grad = ", x_1.grad)
    print("x2.grad = ", x_2.grad)
    print("w1.grad = ", w_1.grad)

    x_1 = Value(data=2.0, label="x1")
    x_2 = Value(data=0.0, label="x2")
    w_1 = Value(data=-3.0, label="w1")
    w_2 = Value(data=1.0, label="w2")
    b = Value(data=6.8813735870195432, label="b")

    logit = get_logit(x_1, x_2, w_1, w_2, b)

    builtin_tanh = logit.tanh()
    builtin_tanh.backward()
    print("o with builtin_tanh = ", builtin_tanh)
    print("x1.grad = ", x_1.grad)
    print("x2.grad = ", x_2.grad)