def create_dataset(n: int) -> list[tuple[int, int]]:
    return [(i, 2*i) for i in range(n)]