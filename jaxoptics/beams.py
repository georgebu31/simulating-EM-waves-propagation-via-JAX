"""Beam profile generation utilities."""

import jax.numpy as jnp


def gauss_profile(rad: float, num: int, res: float):
    """Generate a 2-D Gaussian amplitude profile.

    Args:
        rad:  1/e beam radius [same units as res].
        num:  Grid size (number of pixels per side).
        res:  Pixel pitch [m or any consistent unit].

    Returns:
        (profile, X, Y) — amplitude array and centred coordinate grids.
    """
    x = (jnp.arange(num, dtype=jnp.float32) - (num - 1) / 2) * res
    y = (jnp.arange(num, dtype=jnp.float32) - (num - 1) / 2) * res
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    r = jnp.sqrt(X ** 2 + Y ** 2)
    gaussian = jnp.exp(-(r / rad) ** 2)
    profile = jnp.where(gaussian > 0.01, gaussian, 0.0)
    return profile, X, Y
