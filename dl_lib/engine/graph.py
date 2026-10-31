from graphviz import Digraph

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


def draw_dot(value: Value) -> Digraph:
    dot = Digraph(graph_attr={"rankdir": "LR"})
    nodes, edges = trace(value)

    for node in nodes:
        node_id = str(id(node))
        dot.node(
            node_id,
            label=f"{{ {node.label} | data: {node.data:.4f} | grad: {node.grad:.4f} }}",
            shape="record",
        )
        if node._op:
            op_id = node_id + "_op"
            dot.node(op_id, label=node._op, shape="ellipse")
            dot.edge(op_id, node_id)

    for child, parent in edges:
        dot.edge(str(id(child)), str(id(parent)) + "_op")

    return dot
