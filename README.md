# bilback

Multi-dispatch implementation of GW-specific geometric and time operations for `Bilby` powered by `plum-dispatch`.

This is currently an initial test bed and no stability is guaranteed.

## Installation

`bilback` can be installed as usual for `Python` packages.
By default, only the `numpy` backend is enabled.
Note that for many operations, this is significantly slower than the others.

There is native support for five additional backends

- `bilby_rust` gives CPU-only implementation in `rust`
- `bilby_cython` gives CPU-only implementation in `cython`
- `jax` uses JAX and supports hardware acceleration and just-in-time compilation
- `mlx` uses MLX and supports hardware acceleration and just-in-time compilation,
  however, MLX does not support double-precision floats and so is of limited use
- `cupy` provides GPU-only support

Each of these additional backends can be installed from `PyPI` or `conda-forge`

## Usage

In order to use the functions implemented here you can import them from the

```python
from bilback import geometry
geometry.get_polarization_tensor(ra=0.0, dec=0.0, time=0.0, psi=0.0, mode="plus")
```

The backend used is automatically inferred from the types of the input arguments.
For example, to automatically use the `JAX` backend one simply has to do

```python
import jax.numpy as jnp
from bilback import geometry
geometry.get_polarization_tensor(ra=jax.array(0.0), dec=0.0, time=0.0, psi=0.0, mode="plus")
```
