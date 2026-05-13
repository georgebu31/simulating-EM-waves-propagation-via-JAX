"""Fresnel diffraction propagators (single-step FFT methods).

Two complementary approaches are provided:

1. ``propagate_fresnel``  — Fresnel approximation via a single FFT.
   The output pixel pitch differs from the input::

       dx_out = lambda * z / (N * dx_in)

   Suitable for the **far / intermediate field** where the output
   plane is much smaller than the input aperture.

2. ``propagate_fraunhofer`` — Fraunhofer (far-field) limit, identical
   math to ``propagate_fresnel`` but explicit about the regime and
   includes the full complex prefactor.

Both methods work in the paraxial (small-angle) regime.
"""

from functools import partial
import jax
import jax.numpy as jnp


def fresnel_output_coords(
    num: int, dx_in: float, z: float, lambd: float
) -> jax.Array:
    """Return output-plane coordinates [m] for the Fresnel single-FFT method.

    Args:
        num:    Grid size (pixels per side).
        dx_in:  Input pixel pitch [m].
        z:      Propagation distance [m].
        lambd:  Wavelength [m].

    Returns:
        1-D array of length *num* with centred output coordinates [m].
    """
    dx_out = lambd * z / (num * dx_in)
    coords = (jnp.arange(num) - num // 2) * dx_out
    return coords


@partial(jax.jit, static_argnums=(1,))
def propagate_fresnel(
    E_in: jax.Array,
    num: int,
    dx_in: float,
    z: float,
    lambd: float,
) -> jax.Array:
    """Propagate a complex field using the Fresnel single-FFT method.

    The algorithm evaluates the Fresnel diffraction integral

    .. math::

        U(x', y') = \\frac{e^{ikz}}{i\\lambda z}
            \\iint U_0(x,y)\,
            e^{\\frac{ik}{2z}(x^2+y^2)}\,
            e^{-i\\frac{2\\pi}{\\lambda z}(x x' + y y')} dx\,dy

    as a single 2-D FFT.

    **Output pixel pitch** (different from input)::

        dx_out = lambda * z / (N * dx_in)

    Use :func:`fresnel_output_coords` to obtain the physical coordinates
    of the output plane.

    Args:
        E_in:   Input complex field, shape (N, N).
        num:    Grid size — static JIT arg (must equal E_in.shape[0]).
        dx_in:  Input pixel pitch [m].
        z:      Propagation distance [m].
        lambd:  Wavelength [m].

    Returns:
        E_out: Propagated complex field, shape (N, N), on output-plane
               coordinates given by :func:`fresnel_output_coords`.
    """
    k = 2 * jnp.pi / lambd

    # Input-plane coordinate grid
    x = (jnp.arange(num) - num // 2) * dx_in
    X, Y = jnp.meshgrid(x, x, indexing='ij')

    # Quadratic phase (Fresnel chirp)
    quad = jnp.exp(1j * (k / (2 * z)) * (X ** 2 + Y ** 2))

    # Prefactor (scalar, does not affect intensity pattern)
    prefactor = jnp.exp(1j * k * z) / (1j * lambd * z) * (dx_in ** 2)

    # Single-step FFT
    F = jnp.fft.fftshift(
        jnp.fft.fft2(
            jnp.fft.ifftshift(E_in * quad)
        )
    )
    return prefactor * F


@partial(jax.jit, static_argnums=(1,))
def propagate_fraunhofer(
    E_in: jax.Array,
    num: int,
    dx_in: float,
    z: float,
    lambd: float,
) -> jax.Array:
    """Fraunhofer (far-field) propagation via a single FFT.

    Mathematically identical to :func:`propagate_fresnel`; the quadratic
    input-plane phase is included for completeness but is negligible in
    the strict far field.  The function is provided as a clearly-named
    alias for the far-field regime.

    The validity condition for the Fraunhofer approximation is::

        z >> pi * D^2 / (4 * lambda)

    where *D* is the diameter of the illuminated aperture.

    Args / Returns: identical to :func:`propagate_fresnel`.
    """
    return propagate_fresnel(E_in, num, dx_in, z, lambd)


def fresnel_validity_check(
    num: int,
    dx_in: float,
    z: float,
    lambd: float,
    verbose: bool = True,
) -> dict:
    """Report Fresnel / Fraunhofer regime for given parameters.

    Args:
        num:     Grid size.
        dx_in:   Input pixel pitch [m].
        z:       Propagation distance [m].
        lambd:   Wavelength [m].
        verbose: Print a human-readable report.

    Returns:
        Dict with keys 'dx_out', 'fresnel_number', 'fraunhofer_ok'.
    """
    D = num * dx_in                      # aperture diameter
    dx_out = lambd * z / (num * dx_in)   # output pixel pitch
    N_F = D ** 2 / (lambd * z)           # Fresnel number
    fraunhofer_ok = N_F < 1.0

    if verbose:
        print(f"Wavelength   : {lambd*1e3:.4f} mm")
        print(f"Input  pitch : {dx_in*1e3:.3f} mm,  aperture D = {D*1e3:.1f} mm")
        print(f"Output pitch : {dx_out*1e6:.3f} µm,  FOV = {num*dx_out*1e3:.3f} mm")
        print(f"Fresnel number N_F = D²/(λz) = {N_F:.4f}")
        print(f"  N_F >> 1 : near field  (Fresnel approx valid, Fraunhofer NOT)")
        print(f"  N_F ~  1 : intermediate field")
        print(f"  N_F << 1 : far field   (both Fresnel and Fraunhofer valid)")
        label = '✅ far field' if fraunhofer_ok else '⚠️  near/intermediate field'
        print(f"Regime: {label}")

    return {'dx_out': dx_out, 'fresnel_number': N_F, 'fraunhofer_ok': fraunhofer_ok}
