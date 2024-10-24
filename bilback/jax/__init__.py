import jax
from plum import dispatch

from .time import *


@dispatch
def array_module(arr: jax.Array):
    return jax.numpy