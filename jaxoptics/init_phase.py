"""Phase-mask initialisation heuristics."""

import jax.numpy as jnp
import jax


def make_phi_init(
    X: jax.Array,
    Y: jax.Array,
    centers_m: jax.Array,
    P_target: jax.Array,
    lambd_m: float,
    z_m: float,
    R_threshold_m: float,
    k_grating_x: float | None = None,
):
    """Generate an initial phase mask as a superposition of blazed gratings.

    Inside the beam aperture, the phase is the angle of a coherent sum of
    plane waves steered toward each output port.  Outside the aperture a
    single blazed grating redirects stray light away from the ports.

    Args:
        X, Y:           Centred coordinate grids [m].
        centers_m:      Output port coordinates, shape (N, 2).
        P_target:       Target power weights, shape (N,).
        lambd_m:        Wavelength [m].
        z_m:            Propagation distance [m].
        R_threshold_m:  Aperture radius [m].
        k_grating_x:    Spatial frequency of the outer grating [rad/m].
                        Defaults to 2 × max port k_x.

    Returns:
        (phi_init, aperture_mask) — both of shape (num, num).
    """
    k0 = 2 * jnp.pi / lambd_m
    A = jnp.sqrt(P_target / (jnp.sum(P_target) + 1e-12))
    kn_x = k0 * centers_m[:, 0] / z_m
    kn_y = k0 * centers_m[:, 1] / z_m
    phase_n = kn_x[:, None, None] * X[None] + kn_y[:, None, None] * Y[None]
    field_sum = jnp.sum(A[:, None, None] * jnp.exp(1j * phase_n), axis=0)
    ap = (X ** 2 + Y ** 2 <= R_threshold_m ** 2).astype(jnp.float32)
    if k_grating_x is None:
        k_grating_x = float(2.0 * jnp.max(jnp.abs(kn_x)))
    phi_grating = k_grating_x * X
    field_full = field_sum * ap + jnp.exp(1j * phi_grating) * (1.0 - ap)
    return jnp.angle(field_full).astype(jnp.float32), ap


def phi_init_paper(
    X: jax.Array,
    Y: jax.Array,
    centers_m: jax.Array,
    lambd_m: float,
    z_m: float,
) -> jax.Array:
    """Initial phase following Tian et al. (2023): superposition of blazed gratings.

    Args:
        X, Y:       Centred coordinate grids [m].
        centers_m:  Output port coordinates, shape (N, 2).
        lambd_m:    Wavelength [m].
        z_m:        Propagation distance [m].

    Returns:
        phi_init: phase array of shape (num, num).
    """
    k0 = 2 * jnp.pi / lambd_m
    field = jnp.zeros_like(X, dtype=jnp.complex64)
    for n in range(centers_m.shape[0]):
        cx, cy = centers_m[n, 0], centers_m[n, 1]
        denom = jnp.sqrt(cx ** 2 + cy ** 2 + z_m ** 2)
        kx_n = k0 * cx / denom
        ky_n = k0 * cy / denom
        field = field + jnp.exp(1j * (kx_n * X + ky_n * Y))
    return jnp.angle(field)
