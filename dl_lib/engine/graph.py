from dl_lib.engine.value import Value


def trace(value: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    nodes = set()
    edges = set()

    def visit(current: Value) -> None:
        if current in nodes:
            return

        nodes.add(current)
        for operand in current._prev:
            edges.add((operand, current))
            visit(operand)

    visit(value)

    return (nodes, edges)
