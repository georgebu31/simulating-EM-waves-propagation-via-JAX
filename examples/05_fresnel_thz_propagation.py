"""Example 05 — Fresnel propagation of a THz beam loaded from amplitude/phase maps.

Task parameters (matching a typical university assignment):
    frequency : 700 GHz
    wave speed: 3e8 m/s  (free-space)
    grid step : dx = dy = 12 mm
    grid size : 700 x 700 cells
    distance  : z = 7.2 m
    input     : amplitude map  -> 9_amp.txt
                phase map      -> 9_phase.txt  (radians)

This example uses the Fresnel single-FFT propagator from
``jaxoptics.fresnel`` which is well-suited for THz beams where
the grid step is much larger than the wavelength (dx >> lambda).
In this regime the output plane has a *different* (smaller) pixel
pitch, making the diffracted pattern clearly visible.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxoptics.fresnel import (
    propagate_fresnel,
    fresnel_output_coords,
    fresnel_validity_check,
)
from jaxoptics import intensity

# ─── Parameters ──────────────────────────────────────────────────────────────
freq  = 700e9        # Hz
c     = 3e8          # m/s
lambd = c / freq     # wavelength [m]  ~ 0.4286 mm

dx = 12e-3           # input grid step [m]
N  = 700             # grid size (square)
z  = 7.2             # propagation distance [m]

# ─── Load amplitude and phase maps ───────────────────────────────────────────
amp   = np.loadtxt("9_amp.txt")      # shape (700, 700)
phase = np.loadtxt("9_phase.txt")    # shape (700, 700), radians
# If phase is in degrees, uncomment the next line:
# phase = np.radians(phase)

E_in = jnp.array(amp * np.exp(1j * phase), dtype=jnp.complex64)

# ─── Validity check ──────────────────────────────────────────────────────────
print("=" * 55)
print("Fresnel validity check")
print("=" * 55)
info = fresnel_validity_check(N, dx, z, lambd)
print()

# ─── Propagate ───────────────────────────────────────────────────────────────
E_out = propagate_fresnel(E_in, N, dx, z, lambd)

# ─── Output-plane coordinates ────────────────────────────────────────────────
coords_out = fresnel_output_coords(N, dx, z, lambd)   # [m]
extent_in  = [-N*dx/2*1e3,  N*dx/2*1e3,
              -N*dx/2*1e3,  N*dx/2*1e3]               # mm
extent_out = [float(coords_out[0])*1e3,  float(coords_out[-1])*1e3,
              float(coords_out[0])*1e3,  float(coords_out[-1])*1e3]  # mm

# ─── Plot ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(
    intensity(E_in), cmap='inferno', origin='lower', extent=extent_in
)
axes[0].set_title("Input intensity |E|²")
axes[0].set_xlabel("x, mm"); axes[0].set_ylabel("y, mm")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    intensity(E_out), cmap='inferno', origin='lower', extent=extent_out
)
dx_out_mm = info['dx_out'] * 1e3
axes[1].set_title(
    f"Output intensity |E|²\n"
    f"z = {z} m,  dx_out = {dx_out_mm:.4f} mm\n"
    f"Fresnel N_F = {info['fresnel_number']:.4f}"
)
axes[1].set_xlabel("x', mm"); axes[1].set_ylabel("y', mm")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(
    np.angle(np.array(E_out)), cmap='hsv', origin='lower', extent=extent_out
)
axes[2].set_title("Output phase arg(E)")
axes[2].set_xlabel("x', mm"); axes[2].set_ylabel("y', mm")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(
    f"THz Fresnel propagation  |  f = {freq/1e9:.0f} GHz  |  "
    f"λ = {lambd*1e3:.4f} mm  |  z = {z} m",
    fontsize=13
)
plt.tight_layout()
plt.savefig("05_fresnel_thz_result.png", dpi=150)
plt.show()
print("Saved: 05_fresnel_thz_result.png")
