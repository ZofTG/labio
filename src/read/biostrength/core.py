"""core module"""


#! exports


__all__ = ["G", "polyval", "der1", "der2", "smooth"]


#! constants


G = 9.80665


#! functions


def polyval(
    coefs: list[float | int],
    value: float | int,
):
    """
    apply the polynomial coefficients to a given value.

    Parameters
    ----------
    coefs : list[float  |  int]
        the coefficients of the polynomial (higher order first)

    value : float | int
        the value to be be multiplied by the coefficients

    Returns
    -------
    y: float
        the result of the calculation.
    """
    out = 0
    for i, c in enumerate(coefs):
        out += c * value ** (len(coefs) - i - 1)
    return float(out)


def symmetry(
    val1: float | int,
    val2: float | int,
):
    """
    return the symmetry index between the two values

    Parameters
    ----------
    val1 : list[float  |  int]
        the first array

    val2 : list[float  |  int]
        the second array

    Returns
    -------
    sym: list[float]
        the symmetry at each time instant
    """
    num = abs(val2) - abs(val1)
    den = abs(val1) + abs(val2)
    return 100.0 * (1 if den == 0 else num / den)


def der1(
    y0: float | int,
    y2: float | int,
    x0: float | int,
    x2: float | int,
):
    """
    return the speed value at time instant 1 given the samples at time instant
    0 and 2.

    Parameters
    ----------
    y0: float | int,
        the parameter value at instant 0

    y2: float | int,
        the parameter value at instant 2

    x0: float | int,
        the time instant 0

    x2: float | int,
        the time instant 2

    Returns
    -------
    speed: float
        the first derivative at time instant 1
    """
    num = y2 - y0
    den = x2 - x0
    return 0 if den == 0 else (num / den)


def der2(
    y0: float | int,
    y1: float | int,
    y2: float | int,
    x0: float | int,
    x1: float | int,
    x2: float | int,
):
    """
    return the second derivative at time instant 1 given the samples at time
    instant 0 and 2.

    Parameters
    ----------
    y0: float | int,
        the parameter value at instant 0

    y1: float | int,
        the parameter value at instant 1

    y2: float | int,
        the parameter value at instant 2

    x0: float | int,
        the time instant 0

    x1: float | int
        the time instant 1

    x2: float | int,
        the time instant 2

    Returns
    -------
    acc: float
        the second derivative at time instant 1
    """
    d10 = der1(y0, y1, x0, x1)
    d12 = der1(y1, y2, x1, x2)
    return der1(d10, d12, (x0 + x1) / 2, (x1 + x2) / 2)


def smooth(
    obj: list[float | int],
    order: int = 7,
):
    """
    smooth the signal using a moving average filter

    Parameters
    ----------
    obj : list[float | int]
        the signal to be smoothed

    order : int, optional
        the filter order

    Returns
    -------
    arr: list[float]
        the filtered signal

    Note
    ----
    the order of the filter must be odd.
    """
    init = int(order // 2)
    out = obj.copy()
    for i in range(init, len(out) - init):
        buf = 0
        for j in range(i - init, i + init + 1):
            buf += obj[j]
        out[i] = buf / order
    return list(map(lambda x: round(x, 3), out))
