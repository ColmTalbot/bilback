import numpy as np
from plum import dispatch

from bilby_cython import geometry as _geometry

__all__ = [
    # "antenna_response",
    "calculate_arm",
    "detector_tensor",
    "get_polarization_tensor",
    "get_polarization_tensor_multiple_modes",
    "rotation_matrix_from_delta",
    "three_by_three_matrix_contraction",
    "time_delay_geocentric",
    "time_delay_from_geocenter",
    "zenith_azimuth_to_theta_phi",
]


# @dispatch(precedence=2)
# def antenna_response(detector_tensor: np.ndarray, ra, dec, time, psi, mode):
#     """"""
#     return geometry.antenna_response(detector_tensor, ra, dec, time, psi, mode)


@dispatch(precedence=2)
def calculate_arm(arm_tilt: float, arm_azimuth: float, longitude: float, latitude: float):
    """"""
    return _geometry.calculate_arm(arm_tilt, arm_azimuth, longitude, latitude)


@dispatch(precedence=2)
def detector_tensor(x: np.ndarray, y: np.ndarray):
    """"""
    return _geometry.detector_tensor(x, y)


@dispatch(precedence=2)
def get_polarization_tensor(ra: float, dec: float, time: float, psi: float, mode: str):
    """"""
    return _geometry.get_polarization_tensor(ra, dec, time, psi, mode)


@dispatch(precedence=2)
def get_polarization_tensor_multiple_modes(ra: float, dec: float, time: float, psi: float, modes: list[str]):
    """"""
    return _geometry.get_polarization_tensor_multiple_modes(ra, dec, time, psi, list(modes))


@dispatch(precedence=2)
def rotation_matrix_from_delta(delta: np.ndarray):
    """"""
    return _geometry.rotation_matrix_from_delta(delta)


@dispatch(precedence=2)
def three_by_three_matrix_contraction(a: np.ndarray, b: np.ndarray):
    """"""
    return _geometry.three_by_three_matrix_contraction(a, b)


@dispatch(precedence=2)
def time_delay_geocentric(detector1: np.ndarray, detector2: np.ndarray, ra: float, dec: float, time: float):
    """"""
    return _geometry.time_delay_geocentric(detector1, detector2, ra, dec, time)


@dispatch(precedence=2)
def time_delay_from_geocenter(detector1: np.ndarray, ra: float, dec: float, time: float):
    """"""
    return _geometry.time_delay_from_geocenter(detector1, ra, dec, time)


@dispatch(precedence=2)
def zenith_azimuth_to_theta_phi(zenith: float, azimuth: float, delta_x: np.ndarray):
    """"""
    return _geometry.zenith_azimuth_to_theta_phi(zenith, azimuth, delta_x)
