import mlx.core as mx
from plum import dispatch

from ..time import LEAP_SECONDS as _LEAP_SECONDS, n_leap_seconds

__all__ = ["n_leap_seconds"]

LEAP_SECONDS = mx.array(_LEAP_SECONDS, dtype=mx.int32)


@dispatch
def n_leap_seconds(date: mx.array):
    """
    Find the number of leap seconds required for the specified date.
    """
    return n_leap_seconds(date, LEAP_SECONDS)
