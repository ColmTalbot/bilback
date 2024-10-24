import numpy as np
from plum import dispatch
from bilby_rust import geometry as _geometry

from .types import Real, ArrayLike


__all__ = [
    # "antenna_response",
    "calculate_arm",
    "detector_tensor",
    "get_polarization_tensor",
    # "get_polarization_tensor_multiple_modes",
    "rotation_matrix_from_delta",
    # "three_by_three_matrix_contraction",
    "time_delay_geocentric",
    "time_delay_from_geocenter",
    "zenith_azimuth_to_theta_phi",
]


# @dispatch(precedence=1)
# def antenna_response(detector_tensor: np.ndarray, ra: FloatOrInt, dec: FloatOrInt, time: FloatOrInt, psi: FloatOrInt, mode: str):
#     return _geometry.antenna_response(detector_tensor, ra, dec, time, psi, mode)


@dispatch(precedence=1)
def calculate_arm(arm_tilt: Real, arm_azimuth: Real, longitude: Real, latitude: Real):
    return _geometry.calculate_arm(arm_tilt, arm_azimuth, longitude, latitude)


@dispatch(precedence=1)
def detector_tensor(x: ArrayLike, y: ArrayLike):
    return _geometry.detector_tensor(x, y)


@dispatch(precedence=1)
def get_polarization_tensor(ra: Real, dec: Real, time: Real, psi: Real, mode: str):
    return _geometry.get_polarization_tensor(ra, dec, time, psi, mode)


# @dispatch(precedence=1)
# def get_polarization_tensor_multiple_modes(ra: FloatOrInt, dec: FloatOrInt, time: FloatOrInt, psi: FloatOrInt, modes: list[str]):
#     return [geometry.get_polarization_tensor(ra, dec, time, psi, mode) for mode in modes]


@dispatch(precedence=1)
def rotation_matrix_from_delta(delta: ArrayLike):
    return _geometry.rotation_matrix_from_delta_x(delta)


# @dispatch(precedence=1)
# def three_by_three_matrix_contraction(a: ArrayLike, b: ArrayLike):
#     return _geometry.three_by_three_matrix_contraction(a, b)


@dispatch(precedence=1)
def time_delay_geocentric(detector1: ArrayLike, detector2: ArrayLike, ra, dec, time):
    return _geometry.time_delay_geocentric(detector1, detector2, ra, dec, time)


@dispatch(precedence=1)
def time_delay_from_geocenter(detector1: ArrayLike, ra: Real, dec: Real, time: Real):
    return _geometry.time_delay_from_geocenter(detector1, ra, dec, time)


@dispatch(precedence=1)
def time_delay_from_geocenter(detector1: ArrayLike, ra: Real, dec: Real, time: ArrayLike):
    return _geometry.time_delay_from_geocenter_vectorized(detector1, ra, dec, time)


@dispatch(precedence=1)
def zenith_azimuth_to_theta_phi(zenith: Real, azimuth: Real, delta_x: ArrayLike):
    theta, phi = _geometry.zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
    return theta, phi % (2 * np.pi)
