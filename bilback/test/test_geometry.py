import itertools
import pytest

import bilback
import numpy as np
from bilby.gw.detector import get_empty_interferometer, InterferometerList

from .old_code import (
    antenna_response,
    calculate_arm,
    get_polarization_tensor,
    get_polarization_tensor_multiple_modes,
    time_delay_geocentric,
    zenith_azimuth_to_theta_phi,
)
from ..utils import promote_to_array
from . import maybe_jit

IFOS = ["H1", "L1", "V1", "K1"]
MODES = ["plus", "cross", "x", "y", "breathing", "longitudinal"]

LOOSE_THRESHOLD = {
    "mlx.core": 1e-3,
}
TIGHT_THRESHOLD = {
    "mlx.core": 1e-5,
}


def test_time_delay(backend):
    func = maybe_jit(bilback.geometry.time_delay_geocentric, backend)
    max_diff = 0
    for ifo_pair in itertools.product(IFOS, repeat=2):
        if ifo_pair[0] == ifo_pair[1]:
            continue
        ifos = InterferometerList(ifo_pair)
        detectors = [ifo.vertex for ifo in ifos]

        alt_detectors = [backend.array(ifo.vertex) for ifo in ifos]
        for point in np.random.uniform(0, np.pi / 2, (1000, 3)):
            numpy_delay = time_delay_geocentric(*detectors, *point)
            point = backend.array(point)
            cython_delay = float(func(*alt_detectors, *point))
            max_diff = max(max_diff, abs(numpy_delay - cython_delay))
    assert max_diff < LOOSE_THRESHOLD.get(backend.__name__, 1e-6)


def test_time_delay_from_geocenter_matches_time_delay_geocentric(backend):
    delay_geocentric = maybe_jit(bilback.geometry.time_delay_geocentric, backend)
    time_delay = maybe_jit(bilback.geometry.time_delay_from_geocenter, backend)
    geocenter = backend.zeros(3)
    for ifo in IFOS:
        detector = backend.array(get_empty_interferometer(ifo).vertex)
        for point in np.random.uniform(0, np.pi / 2, (1000, 3)):
            point = backend.array(point)
            geocentric = delay_geocentric(detector, geocenter, *point)
            from_geocenter = time_delay(detector, *point)
            assert abs(from_geocenter - geocentric) < TIGHT_THRESHOLD.get(backend.__name__, 1e-10)


def test_get_polarization_tensor(backend):
    func = maybe_jit(bilback.geometry.get_polarization_tensor, backend, static_argnames=("mode",))
    max_diff = 0
    for ra, dec, time, psi in np.random.uniform(0, np.pi / 2, (100, 4)):
        for mode in MODES:
            args = (ra, dec, time, psi, mode)
            numpy_tensor = get_polarization_tensor(*args)
            args = promote_to_array(args, backend, skip=1)
            cython_tensor = func(*args)
            max_diff = abs(np.max(numpy_tensor - cython_tensor))
    assert max_diff < TIGHT_THRESHOLD.get(backend.__name__, 1e-8)


def test_get_polarization_tensor_multiple_modes(backend):
    func = maybe_jit(bilback.geometry.get_polarization_tensor_multiple_modes, backend, static_argnames=("modes",))
    modes = tuple(MODES)
    max_diff = 0
    for ra, dec, time, psi in np.random.uniform(0, np.pi / 2, (100, 4)):
        args = (ra, dec, time, psi, modes)
        numpy_tensors = get_polarization_tensor_multiple_modes(*args)
        args = promote_to_array(args, backend, skip=1)
        cython_tensors = func(*args)
        max_diff = abs(np.max(np.array(numpy_tensors) - np.array(cython_tensors)))
    assert max_diff < TIGHT_THRESHOLD.get(backend.__name__, 1e-8)


def test_polarization_tensor_bad_mode_raises_error():
    with pytest.raises(ValueError):
        _ = bilback.geometry.get_polarization_tensor(
            0.0, 0.0, 0.0, 0.0, "bad_mode"
        )


def test_antenna_response(backend):
    func = maybe_jit(bilback.geometry.antenna_response, backend, static_argnames=("mode",))
    max_diff = 0
    for ifo in IFOS:
        # we have to explicitly cast to numpy as rapid switching breaks cases
        # where functions have been imported previously
        detector = np.array(get_empty_interferometer(ifo).geometry.detector_tensor)
        for ra, dec, time, psi in np.random.uniform(0, np.pi / 2, (100, 4)):
            for mode in MODES:
                args = (ra, dec, time, psi, mode)
                numpy_tensor = antenna_response(detector, *args)
                args = promote_to_array(args, backend, skip=1)
                cython_tensor = func(backend.array(detector), *args)
                max_diff = abs(np.max(numpy_tensor - cython_tensor))
    assert max_diff < LOOSE_THRESHOLD.get(backend.__name__, 1e-7)


def test_frame_conversion(backend):
    func = maybe_jit(bilback.geometry.zenith_azimuth_to_theta_phi, backend)
    max_diff = 0
    for ifo_pair in itertools.product(IFOS, repeat=2):
        if ifo_pair[0] == ifo_pair[1]:
            continue
        ifos = InterferometerList(ifo_pair)
        delta_x = ifos[0].vertex - ifos[1].vertex
        # there is a bug in the python implementation for separations
        # lying in specific quadrants for arctan2 due to using arctan
        # the direct tests of the rotation matrix are more rigorous
        sign_incorrect = abs(np.arctan2(delta_x[1], delta_x[0])) < np.pi / 2
        delta_x *= (-1) ** sign_incorrect
        for point in np.random.uniform(0, np.pi / 2, (100, 2)):
            numpy_result = zenith_azimuth_to_theta_phi(*point, delta_x)
            cython_result = func(*backend.array(point), backend.array(delta_x))
            max_diff = max(
                max_diff, np.max(abs(np.array(numpy_result) - np.array(cython_result)))
            )
    assert max_diff < LOOSE_THRESHOLD.get(backend.__name__, 1e-6)


def test_calculate_arm(backend):
    func = maybe_jit(bilback.geometry.calculate_arm, backend)
    max_diff = 0
    for point in np.random.uniform(0, 2 * np.pi, (1000, 4)):
        max_diff = max(
            max_diff,
            np.linalg.norm(func(*backend.array(point)) - calculate_arm(*point)),
        )
    assert max_diff < TIGHT_THRESHOLD.get(backend.__name__, 1e-10)


def test_detector_tensor(backend):
    func = maybe_jit(bilback.geometry.detector_tensor, backend)
    for xx, yy in np.random.uniform(0, 1, (1000, 2, 3)):
        numpy_tensor = 0.5 * (
            np.einsum("i,j->ij", xx, xx) - np.einsum("i,j->ij", yy, yy)
        )
        cython_tensor = func(backend.array(xx), backend.array(yy))
        assert np.max(np.abs(numpy_tensor - cython_tensor)) < 1e-6


def test_rotation_matrix_transpose_is_inverse(backend):
    func = maybe_jit(bilback.geometry.rotation_matrix_from_delta, backend)
    for delta_x in np.random.uniform(0, 1, (100, 3)):
        rotation = func(backend.array(delta_x))
        assert np.max(np.abs((rotation.T @ rotation - np.eye(3)))) < TIGHT_THRESHOLD.get(backend.__name__, 1e-10)


def test_rotation_matrix_maps_delta_x_to_z_axis(backend):
    func = maybe_jit(bilback.geometry.rotation_matrix_from_delta, backend)
    for delta_x in np.random.uniform(0, 1, (100, 3)):
        rotation = np.asarray(func(backend.array(delta_x)))
        delta_x /= np.linalg.norm(delta_x)
        assert np.max(np.abs(rotation.T @ delta_x - np.array([0, 0, 1]))) < TIGHT_THRESHOLD.get(backend.__name__, 1e-10)
