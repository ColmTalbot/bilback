from typing import Union

import numpy as np
from plum import dispatch

__all__ = ["array_module", "promote_to_array"]


@dispatch
def array_module(arr: Union[list, tuple, float, int, np.ndarray]):
    return np


def promote_to_array(args, backend, skip=None):
    if skip is None:
        skip = len(args)
    else:
        skip = len(args) - skip
    if backend.__name__ != "numpy":
        args = tuple(backend.array(arg) for arg in args[:skip]) + args[skip:]
    return args
