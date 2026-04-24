"""Intensity computation."""

import jax
import jax.numpy as jnp


@jax.jit
def intensity(E: jax.Array) -> jax.Array:
    """Compute optical intensity |E|^2 from a complex field."""
    return jnp.abs(E) ** 2
