import numpy as np
from plum import dispatch

from bilby_cython import _time


@dispatch(precedence=2)
def gps_time_to_utc(gps_time):
    """"""
    return _time.gps_time_to_utc(gps_time)


@dispatch(precedence=2)
def greenwich_mean_sidereal_time(gps_time):
    """"""
    return _time.greenwich_mean_sidereal_time(gps_time)


@dispatch(precedence=2)
def greenwich_sidereal_time(gps_time, equation_of_equinoxes):
    """"""
    return _time.greenwich_sidereal_time(gps_time, equation_of_equinoxes)


@dispatch(precedence=2)
def n_leap_seconds(gps_time):
    """"""
    return _time.n_leap_seconds(gps_time)


@dispatch(precedence=2)
def utc_to_julian_day(utc_time):
    """"""
    return _time.utc_to_julian_day(utc_time)
