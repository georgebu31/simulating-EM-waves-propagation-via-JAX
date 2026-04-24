"""Phase manipulation utilities."""

import jax
import jax.numpy as jnp


@jax.jit
def phaseshiftt(phase: jax.Array, prof: jax.Array) -> jax.Array:
    """Apply a real phase map to a real amplitude profile.

    Returns a complex field: prof * exp(i * phase).
    Uses ``jax.lax.complex`` to avoid implicit dtype promotion.
    """
    return prof * jnp.exp(jax.lax.complex(0.0, 1.0) * phase)


@jax.jit
def phaseshift(phase: jax.Array, amp: jax.Array) -> jax.Array:
    """Apply a real phase map to a complex (or real) amplitude.

    Returns: amp * exp(i * phase).
    """
    return amp * jnp.exp(1j * phase)


@jax.jit
def wrap_to_pi(phi: jax.Array) -> jax.Array:
    """Wrap phase values into [-pi, pi]."""
    return jnp.mod(phi + jnp.pi, 2 * jnp.pi) - jnp.pi
