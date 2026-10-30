class Value:
    def __init__(self, data: float):
        self.data = data

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Value(data=(self.data + other))
        elif isinstance(other, Value):
            return Value(data=(self.data + other.data))
        else:
            return NotImplemented

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Value(data=(self.data * other))
        elif isinstance(other, Value):
            return Value(data=(self.data * other.data))
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self * other

