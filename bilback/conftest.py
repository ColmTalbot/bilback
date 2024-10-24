import pytest

from . import BACKENDS


@pytest.fixture(params=list(BACKENDS.keys()))
def backend(request):
    if request.param == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        xp = jax.numpy
    elif request.param == "mlx":
        import mlx

        xp = mlx.core
    else:
        import numpy as xp
    return xp

