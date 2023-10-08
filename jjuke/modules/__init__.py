from inspect import isfunction


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        # return itself when it is already tuple
        return t
    # return converted tuple given element and length
    return ((t,) * length)