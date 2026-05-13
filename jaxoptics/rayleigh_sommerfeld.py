"""Rayleigh-Sommerfeld (RS) diffraction propagator.

Implements the first Rayleigh-Sommerfeld solution via the
convolution theorem (FFT-based), which is exact within scalar
diffraction theory (no paraxial approximation).

The impulse response (Green's function) of free-space propagation is::

    h(x, y; z) = (z / r) * (1/(2*pi)) * exp(i*k*r) * (1/r - i*k) / r

where r = sqrt(x^2 + y^2 + z^2).  In the far field (k*r >> 1) this
simplifies to the familiar form::

    h(x, y; z) ~ (z / (2*pi)) * (i*k / r^2) * exp(i*k*r)

Propagation is evaluated as a 2-D convolution::

    E_out(x', y') = E_in(x, y)  *  h(x-x', y-y'; z)

using the FFT convolution theorem on a zero-padded grid to avoid
circular-convolution artefacts.

Notes
-----
* No paraxial / small-angle approximation is made.
* Evanescent components (kx^2 + ky^2 > k^2) are naturally suppressed
  by the real-space kernel; no explicit masking is needed.
* For very large grids or many z-slices, the ASM approach in
  ``propagation.py`` is faster.  RS shines when dx ~ lambda or when
  high-NA accuracy is required.
* Zero-padding factor 2 is the minimum to avoid wrap-around; increase
  it if the beam is very wide relative to the grid.
"""

from functools import partial
import jax
import jax.numpy as jnp


def _rs_kernel(
    num: int,
    dx: float,
    z: float,
    lambd: float,
    pad: int,
) -> jax.Array:
    """Build the RS impulse-response kernel on a padded grid.

    The kernel is evaluated on a grid of size (num*pad) x (num*pad)
    centred at (0, 0) and returned in *standard* (un-shifted) FFT order
    so it can be multiplied directly with fft2(E_padded).

    Args:
        num:    Original grid size (pixels per side).
        dx:     Pixel pitch [m].
        z:      Propagation distance [m].
        lambd:  Wavelength [m].
        pad:    Zero-padding factor (padded grid = num*pad).

    Returns:
        H: FFT of the RS kernel, shape (num*pad, num*pad), complex.
    """
    k  = 2 * jnp.pi / lambd
    Np = num * pad

    # Coordinate grid centred at 0, in fftshift order
    idx = jnp.fft.ifftshift(jnp.arange(Np) - Np // 2)
    xi  = idx * dx
    XI, ETA = jnp.meshgrid(xi, xi, indexing='ij')

    r = jnp.sqrt(XI**2 + ETA**2 + z**2)

    # First RS solution kernel (exact scalar diffraction)
    # h = (z / r) * exp(ikr) / (2*pi) * (ik - 1/r) / r
    # simplified: h = z * exp(ikr) * (ik*r - 1) / (2*pi * r^3)
    h = (z / (2 * jnp.pi)) * jnp.exp(1j * k * r) * (1j * k - 1.0 / r) / r**2

    return jnp.fft.fft2(h) * (dx**2)


@partial(jax.jit, static_argnums=(1, 2, 3))
def propagate_rs(
    E_in: jax.Array,
    num: int,
    pad: int,
    dx: float,
    z: float,
    lambd: float,
) -> jax.Array:
    """Propagate a complex field using the Rayleigh-Sommerfeld method.

    Uses zero-padding to eliminate circular-convolution artefacts.
    The output field is returned on the **same grid and pixel pitch**
    as the input (the padded region is cropped away).

    Args:
        E_in:   Input complex field, shape (N, N).
        num:    Grid size N — static JIT arg.
        pad:    Zero-padding factor — static JIT arg.
                  pad=1 : no padding (circular convolution, fast)
                  pad=2 : recommended minimum (linear convolution)
        dx:     Pixel pitch [m].
        z:      Propagation distance [m].
        lambd:  Wavelength [m].

    Returns:
        E_out: Propagated complex field, shape (N, N).
    """
    Np = num * pad
    s  = (Np - num) // 2  # start index of the unpadded region

    # Zero-pad input
    E_pad = jnp.zeros((Np, Np), dtype=jnp.complex64)
    E_pad = E_pad.at[s:s + num, s:s + num].set(E_in)

    # Build RS kernel in frequency domain
    H = _rs_kernel(num, dx, z, lambd, pad)

    # Convolve via FFT
    E_out_pad = jnp.fft.ifft2(jnp.fft.fft2(E_pad) * H)

    # Crop back to original size
    return E_out_pad[s:s + num, s:s + num]


def rs_validity_info(
    num: int,
    dx: float,
    z: float,
    lambd: float,
    pad: int = 2,
    verbose: bool = True,
) -> dict:
    """Print regime information for RS propagation.

    Args:
        num:     Grid size.
        dx:      Pixel pitch [m].
        z:       Propagation distance [m].
        lambd:   Wavelength [m].
        pad:     Zero-padding factor.
        verbose: Print report.

    Returns:
        Dict with keys 'fresnel_number', 'dx_over_lambda', 'na_max'.
    """
    D       = num * dx
    N_F     = D**2 / (lambd * z)
    dx_lam  = dx / lambd
    na_max  = jnp.sqrt(1 - (lambd / (2 * dx))**2) if lambd < 2*dx else 0.0

    if verbose:
        print(f"Wavelength     : {lambd*1e3:.4f} mm")
        print(f"dx / lambda    : {float(dx_lam):.2f}  "
              f"({'paraxial OK' if dx_lam > 1 else 'sub-wavelength, high accuracy'})")
        print(f"Aperture D     : {D:.3f} m = {D*1e3:.1f} mm")
        print(f"Fresnel number : {N_F:.4f}")
        print(f"Max NA         : {float(na_max):.4f}")
        print(f"Pad factor     : {pad}  (padded grid {num*pad} x {num*pad})")
        print(f"RS validity    : ✅ always valid (no paraxial approx)")

    return {'fresnel_number': float(N_F),
            'dx_over_lambda': float(dx_lam),
            'na_max': float(na_max)}
