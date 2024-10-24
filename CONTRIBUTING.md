# Contributing Guidelines

Bugfixes, performance improvements, or documentation improvements are always welcome!
Please follow a standard issue/fork and pull mechanism.

## Should I add a function to `bilback`?

This package is intended to provide optimized implementations of functions
specific to gravitaitonal-wave data analysis in `Bilby`.
As one of the core driving principles of `Bilby` is to write software that
is flexible and simple for new users to understand and develop into the
decision to move functions to this optimized package must have a high
barrier.
As a general metric, one should show that the function being ported is:
- taking `O(%)` of the total run time for a not-too-contrived run configuration. 
  This can be verified using the
  `Python` profiler `cProfile` (see, e.g.,
  [this SO](https://stackoverflow.com/questions/582336/how-can-you-profile-a-python-script)
  for more details.)

As an example, the primary motivation for beginning this package was that when
using a very fast source model evaluating the response of the gravitational-wave
detectors contributed a significant amount of run time due to having small
array operations being not very well optimized in `numpy`.
By hard-coding 3x3 matrix operations a significant acceleration is possible.

## How do I add a new backend?

We use a plugin system using [Python entry points](https://setuptools.pypa.io/en/latest/userguide/entry_point.html).
For example to add a JAX new plugin (this is already implemented natively, but the same will apply to any numpy-like
backend) add the following to `pyproject.toml`.

```python
[project.entry-points."bilbackends"]
jax = "bilback.jax"
```

All of the operations have a base `numpy` implementation that can be trivially adapted for numpy-like APIs.
The minimal definition is then

```python
import jax
from bilback.utils import array_module
from plum import dispatch

@dispatch
def array_module(arr: jax.array):
  return jax.numpy
```

You can additionally define custom versions of any of the implemented functions using the `plum`

```python
import jax
import jax.numpy as np
from bilback.utils import array_module
from plum import dispatch

@dispatch
def detector_tensor(x: jax.array, y: jax.array):
    return (xp.einsum("i,j->ij", x, x) - xp.einsum("i,j->ij", y, y)) / 2
```
