import bilback
import lal
import numpy as np
import pytest
from astropy.time import Time

from ..utils import promote_to_array
from . import maybe_jit

EPSILON = {
    "numpy": 1e-10,
    "jax.numpy": 1e-8,
    "mlx.core": 1e-4,
}


def test_gmst(backend):
    if backend.__name__ == "mlx.core":
        pytest.skip("double precision needed for gmst calculations")
    func = maybe_jit(bilback.time.greenwich_mean_sidereal_time, backend)
    times = np.random.uniform(1325623903, 1345623903, 100000)
    alt_times = backend.array(times)
    diffs = list()
    for tt, at in zip(times, alt_times):
        diffs.append(func(at) - lal.GreenwichMeanSiderealTime(tt))
    assert max(np.abs(diffs)) < EPSILON[backend.__name__]


def test_gmst_vectorized(backend):
    if backend.__name__ == "mlx.core":
        pytest.skip("double precision needed for gmst calculations")
    func = maybe_jit(bilback.time.greenwich_mean_sidereal_time, backend)
    times = np.random.uniform(1325623903, 1345623903, 100000)
    cy_gmst = func(backend.array(times))
    lal_gmst = np.array([lal.GreenwichMeanSiderealTime(tt) for tt in times])
    assert max(np.abs(cy_gmst - lal_gmst)) < EPSILON[backend.__name__]


def test_gmt(backend):
    if backend.__name__ == "mlx.core":
        pytest.skip("double precision needed for gmst calculations")
    func = maybe_jit(bilback.time.greenwich_sidereal_time, backend)
    times = np.random.uniform(1325623903, 1345623903, 100000)
    equinoxes = np.random.uniform(0, 2 * np.pi, 100000)
    diffs = list()
    alt_times = backend.array(times)
    alt_eqs = backend.array(equinoxes)
    for tt, eq, at, aq in zip(times, equinoxes, alt_times, alt_eqs):
        diffs.append(float(func(at, aq)) - lal.GreenwichSiderealTime(tt, eq))
    assert max(np.abs(diffs)) < EPSILON[backend.__name__]


def test_current_time(backend):
    """
    Test that the current GMST matches LAL and Astropy.
    This should ensure robustness against additional leap seconds being added.
    """
    func = maybe_jit(bilback.time.greenwich_mean_sidereal_time, backend)
    now = float(lal.GPSTimeNow())
    lal_now = lal.GreenwichMeanSiderealTime(now) % (2 * np.pi)
    args = promote_to_array((now,), backend)
    cython_now = func(*args) % (2 * np.pi)
    astropy_now = Time(now, format="gps").sidereal_time("mean", 0.0).radian
    assert np.abs(cython_now - lal_now) < EPSILON[backend.__name__]
    assert np.abs(cython_now - astropy_now) < max(EPSILON[backend.__name__], 1e-5)


def test_datetime_repr():
    """
    Test that the minimal datetime implementation repr works as expected.
    """
    from datetime import datetime as reference
    from bilback.time import datetime as test

    assert (
        reference(2021, 3, 1, 11, 23, 5).strftime("%Y-%-m-%-d %-H:%-M:%-S")
        == test(2021, 3, 1, 11, 23, 5).__repr__()
    )
