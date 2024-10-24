def maybe_jit(func, backend, **kwargs):
    if backend.__name__ == "jax.numpy":
        import jax
        return jax.jit(func, **kwargs)
    return func