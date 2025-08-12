def to_tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)
