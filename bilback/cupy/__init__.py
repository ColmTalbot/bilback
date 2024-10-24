import cupy as cp
from plum import dispatch

from .time import *


@dispatch
def array_module(arr: cp.array):
    return cp
