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


def nyquist_check(
    num_in: int,
    dx_m: float,
    z_m: float,
    lambd_m: float,
    pad_factor: int = 1,
    verbose: bool = True,
) -> bool:
    """Check the Nyquist sampling condition for ASM propagation.

    The output field after propagating distance *z* has a spatial bandwidth
    that grows as the beam diffracts.  The condition

        dx <= lambda * z / (N_padded * dx)

    must hold to avoid aliasing in the output plane.

    Args:
        num_in:     Input grid size (pixels per side).
        dx_m:       Pixel pitch [m].
        z_m:        Propagation distance [m].
        lambd_m:    Wavelength [m].
        pad_factor: Zero-padding multiplier (output grid = num_in * pad_factor).
        verbose:    Print a human-readable report.

    Returns:
        True if the condition is satisfied, False otherwise.
    """
    N_pad   = num_in * pad_factor
    fov_out = N_pad * dx_m                       # physical output FOV [m]
    # minimum output FOV needed so that diffracted beam fits:
    # w(z) ~ lambda*z / (pi*dx)  (far-field divergence limit)
    fov_min = lambd_m * z_m / dx_m               # Nyquist FOV requirement [m]
    ok      = fov_out >= fov_min
    if verbose:
        print(f"Input  grid : {num_in} × {num_in},  dx = {dx_m*1e6:.3f} µm")
        print(f"Pad factor  : {pad_factor}  →  padded grid {N_pad} × {N_pad}")
        print(f"Output FOV  : {fov_out*1e3:.3f} mm")
        print(f"Min FOV req : {fov_min*1e3:.3f} mm  (λz/dx)")
        print(f"Nyquist OK  : {'✅ YES' if ok else '❌ NO — increase pad_factor'}")
        if not ok:
            needed = int(fov_min / (num_in * dx_m)) + 1
            print(f"  → suggested pad_factor >= {needed}")
    return ok


@partial(jax.jit, static_argnums=(1, 2))
def propagate_asm_padded(
    E_in: jax.Array,
    pad_factor: int,
    dx_m: float,
    z_m: float,
    lambd_m: float,
) -> jax.Array:
    """ASM propagation with zero-padding for an enlarged output plane.

    The input field is embedded in the centre of a zero-padded array of size
    ``(N * pad_factor) × (N * pad_factor)``.  The pixel pitch *dx_m* is
    **unchanged**, so the output field covers a FOV that is ``pad_factor``
    times larger than the input.

    Physical interpretation
    -----------------------
    Zero-padding in the spatial domain is equivalent to denser sampling in
    the frequency domain (more k-space points at the same bandwidth).  The
    propagated field is therefore evaluated on a finer-sampled, wider output
    grid — exactly what you want when the beam diverges significantly.

    Sampling condition (Nyquist)
    ----------------------------
    To avoid aliasing the output FOV must satisfy::

        N_padded * dx_m  >=  lambda * z / dx_m

    Use :func:`nyquist_check` to verify before calling this function.

    Args:
        E_in:       Complex input field, shape (N, N).
        pad_factor: Integer multiplier for the output grid size — static arg.
        dx_m:       Pixel pitch [m] — static arg.
        z_m:        Propagation distance [m].
        lambd_m:    Wavelength [m].

    Returns:
        E_out: Propagated complex field, shape (N*pad_factor, N*pad_factor),
               same pixel pitch *dx_m*, larger FOV.
    """
    N   = E_in.shape[0]
    Np  = N * pad_factor
    s   = (Np - N) // 2

    # embed input in centre of zero-padded grid
    E_pad = jnp.zeros((Np, Np), dtype=E_in.dtype)
    E_pad = E_pad.at[s : s + N, s : s + N].set(E_in)

    H = make_transfer_func(Np, dx_m, z_m, lambd_m)
    return propagate_asm(E_pad, H)
