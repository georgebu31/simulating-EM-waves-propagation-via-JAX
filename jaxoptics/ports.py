"""Output port utilities: target intensity, masks, power, and loss."""

from functools import partial
import jax
import jax.numpy as jnp


def circle_centers_m(n_ports: int, radius_m: float) -> jax.Array:
    """Return (n_ports, 2) array of port centre coordinates on a circle.

    Args:
        n_ports:   Number of output ports.
        radius_m:  Circle radius [m].

    Returns:
        Array of shape (n_ports, 2) with columns [x, y] in metres.
    """
    angles = jnp.linspace(0.0, 2 * jnp.pi, n_ports, endpoint=False, dtype=jnp.float32)
    return jnp.stack([radius_m * jnp.cos(angles), radius_m * jnp.sin(angles)], axis=-1)


def target_intensity_xy(
    X: jax.Array,
    Y: jax.Array,
    centers_m: jax.Array,
    waist_m: float,
    power: jax.Array,
) -> jax.Array:
    """Build a target intensity pattern as a sum of Gaussians at port centres.

    Args:
        X, Y:       Coordinate grids [m].
        centers_m:  Port centres, shape (N, 2).
        waist_m:    1/e^2 beam radius for each spot [m].
        power:      Relative power weights, shape (N,).

    Returns:
        I_target array of shape (num, num).
    """
    w2 = waist_m * waist_m

    def body(i, acc):
        cx, cy = centers_m[i, 0], centers_m[i, 1]
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        return acc + power[i] * jnp.exp(-2.0 * r2 / w2)

    return jax.lax.fori_loop(0, centers_m.shape[0], body, jnp.zeros_like(X, dtype=jnp.float32))


@partial(jax.jit, static_argnums=(3,))
def make_port_masks(
    X: jax.Array,
    Y: jax.Array,
    centers_m: jax.Array,
    sigma_m: float,
    normalize_each: bool = True,
) -> jax.Array:
    """Create Gaussian integration masks for each output port.

    Args:
        X, Y:          Coordinate grids [m].
        centers_m:     Port centres, shape (N, 2).
        sigma_m:       Gaussian sigma for the mask [m] — static arg.
        normalize_each: Normalise each mask to unit sum if True.

    Returns:
        masks: shape (N, num, num).
    """
    s2 = sigma_m * sigma_m
    n_ports = centers_m.shape[0]

    def one_mask(i):
        cx, cy = centers_m[i, 0], centers_m[i, 1]
        r2 = (X - cx) ** 2 + (Y - cy) ** 2
        M = jnp.exp(-0.5 * r2 / s2)
        if normalize_each:
            M = M / (jnp.sum(M) + 1e-12)
        return M.astype(jnp.float32)

    return jax.vmap(one_mask)(jnp.arange(n_ports))


@jax.jit
def port_powers(I: jax.Array, masks: jax.Array) -> jax.Array:
    """Integrate intensity over each port mask.

    Args:
        I:     Intensity array, shape (num, num).
        masks: Mask stack from :func:`make_port_masks`, shape (N, num, num).

    Returns:
        P: power in each port, shape (N,).
    """
    return jnp.sum(masks * I[None, :, :], axis=(1, 2))


@jax.jit
def normalize_ratios(P: jax.Array) -> jax.Array:
    """Normalise a power vector to unit sum."""
    return P / (jnp.sum(P) + 1e-12)


@jax.jit
def ratio_loss(P_out: jax.Array, P_target: jax.Array) -> jax.Array:
    """MSE loss between normalised output and target power ratios."""
    return jnp.mean((normalize_ratios(P_out) - normalize_ratios(P_target)) ** 2)
