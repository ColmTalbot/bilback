import mlx.core as mx
from plum import dispatch

from .time import *


@dispatch
def array_module(arr: mx.array):
    return mx
