from plum import dispatch
from bilby_rust import time as _time

from .types import Real, ArrayLike

__all__ = [
    "gps_time_to_utc",
    "greenwich_mean_sidereal_time",
    "greenwich_sidereal_time",
    "n_leap_seconds",
    "utc_to_julian_day",
]


@dispatch(precedence=1)
def gps_time_to_utc(gps_time: Real):
    return _time.gps_time_to_utc(gps_time)


@dispatch(precedence=1)
def greenwich_mean_sidereal_time(gps_time: Real):
    return _time.greenwich_mean_sidereal_time(gps_time)


@dispatch(precedence=1)
def greenwich_mean_sidereal_time(gps_time: ArrayLike):
    return _time.greenwich_mean_sidereal_time_vectorized(gps_time)


@dispatch(precedence=1)
def greenwich_sidereal_time(gps_time: Real, equation_of_equinoxes: Real):
    return _time.greenwich_sidereal_time(gps_time, equation_of_equinoxes)


@dispatch(precedence=1)
def n_leap_seconds(gps_time: Real):
    return _time.n_leap_seconds(gps_time)


@dispatch(precedence=1)
def utc_to_julian_day(utc_time: Real):
    return _time.utc_to_julian_day(utc_time)
