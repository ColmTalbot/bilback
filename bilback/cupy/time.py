import cupy as cp
from plum import dispatch

from ..time import LEAP_SECONDS as _LEAP_SECONDS, n_leap_seconds

__all__ = ["n_leap_seconds"]

LEAP_SECONDS = cp.array(_LEAP_SECONDS)


@dispatch
def n_leap_seconds(date: cp.array):
    """
    Find the number of leap seconds required for the specified date.
    """
    return n_leap_seconds(date, LEAP_SECONDS)
