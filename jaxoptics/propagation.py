"""Angular Spectrum Method (ASM) propagation."""

from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def make_transfer_func(num: int, dx_m: float, z_m: float, lambd_m: float) -> jax.Array:
    """Compute the ASM transfer function H for free-space propagation.

    Args:
        num:      Grid size (pixels per side) — static (JIT compile-time).
        dx_m:     Pixel pitch [m].
        z_m:      Propagation distance [m].
        lambd_m:  Wavelength [m].

    Returns:
        H: complex transfer function of shape (num, num).
    """
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(num, d=dx_m))
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(num, d=dx_m))
    FX, FY = jnp.meshgrid(fx, fy, indexing="ij")
    kx = 2 * jnp.pi * FX
    ky = 2 * jnp.pi * FY
    k = 2 * jnp.pi / lambd_m
    kz_sq = k * k - (kx * kx + ky * ky)
    kz = jnp.sqrt(kz_sq + 0j)
    return jnp.exp(1j * kz * z_m)


@jax.jit
def propagate_asm(E_in: jax.Array, H: jax.Array) -> jax.Array:
    """Propagate a complex field using the Angular Spectrum Method.

    Args:
        E_in: Input complex field, shape (N, N).
        H:    Transfer function from :func:`make_transfer_func`.

    Returns:
        E_out: Propagated complex field, shape (N, N).
    """
    F = jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(E_in), norm='ortho'))
    E_out = jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(F * H), norm='ortho'))
    return E_out
