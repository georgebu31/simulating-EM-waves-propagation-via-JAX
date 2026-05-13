"""Example 06 — Rayleigh-Sommerfeld propagation of a THz beam.

Task parameters:
    frequency : 700 GHz
    wave speed: 3e8 m/s
    grid step : dx = dy = 12 mm
    grid size : 700 x 700
    distance  : z = 7.2 m
    input     : 9_amp.txt, 9_phase.txt

The Rayleigh-Sommerfeld method makes no paraxial approximation and
is valid for any dx/lambda ratio.  Propagation is implemented as a
2-D convolution with the exact free-space Green's function.

Note on output coordinates
--------------------------
RS preserves the pixel pitch (dx_out = dx_in = 12 mm), so the output
plane has the same spatial extent as the input.  To see a diffracted
pattern that differs visibly from the input, the beam must have
features comparable to the Fresnel zone size:

    r_F = sqrt(lambda * z) ~ sqrt(0.4286e-3 * 7.2) ~ 55.6 mm ~ 4-5 pixels

If the input beam is smooth at this scale, the output will look
similar.  Use the Fresnel propagator (examples/05) to see the
far-field diffraction pattern on a finer output grid.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from jaxoptics.rayleigh_sommerfeld import propagate_rs, rs_validity_info
from jaxoptics import intensity

# ─── Parameters ─────────────────────────────────────────────────────────────
freq  = 700e9
c     = 3e8
lambd = c / freq      # ~0.4286 mm

dx  = 12e-3           # [m]
N   = 700
z   = 7.2             # [m]
PAD = 2               # zero-padding factor (2 = minimum for linear convolution)

# ─── Load data ──────────────────────────────────────────────────────────────────
amp   = np.loadtxt("9_amp.txt")
phase = np.loadtxt("9_phase.txt")   # radians
# phase = np.radians(phase)          # uncomment if degrees

E_in = jnp.array(amp * np.exp(1j * phase), dtype=jnp.complex64)

# ─── Validity / regime info ──────────────────────────────────────────────────────
print("=" * 55)
print("Rayleigh-Sommerfeld propagation: regime info")
print("=" * 55)
info = rs_validity_info(N, dx, z, lambd, pad=PAD)
print()

# ─── Propagate ───────────────────────────────────────────────────────────────
print("Propagating... (first call triggers JIT compilation)")
E_out = propagate_rs(E_in, N, PAD, dx, z, lambd)
print("Done.")
print(f"Output field shape: {E_out.shape}")
print()

# ─── Intensity and phase ────────────────────────────────────────────────────────
I_in  = np.array(intensity(E_in))
I_out = np.array(intensity(E_out))
phi_out = np.angle(np.array(E_out))

extent = [-N*dx/2*1e3, N*dx/2*1e3, -N*dx/2*1e3, N*dx/2*1e3]   # mm

print(f"Input  total power : {I_in.sum():.4e}")
print(f"Output total power : {I_out.sum():.4e}")
print(f"Power ratio        : {I_out.sum()/I_in.sum():.6f}  (should be ~1.0)")

# ─── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

im0 = axes[0].imshow(I_in,  cmap='inferno', origin='lower', extent=extent)
axes[0].set_title("Input intensity |E|²")
axes[0].set_xlabel("x, mm"); axes[0].set_ylabel("y, mm")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(I_out, cmap='inferno', origin='lower', extent=extent)
axes[1].set_title(
    f"RS output intensity |E|²\n"
    f"z = {z} m,  dx = {dx*1e3:.0f} mm\n"
    f"Fresnel N_F = {info['fresnel_number']:.4f}"
)
axes[1].set_xlabel("x, mm"); axes[1].set_ylabel("y, mm")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(phi_out, cmap='hsv', origin='lower', extent=extent)
axes[2].set_title("Output phase arg(E)")
axes[2].set_xlabel("x, mm"); axes[2].set_ylabel("y, mm")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(
    f"Rayleigh-Sommerfeld THz propagation  |  "
    f"f = {freq/1e9:.0f} GHz  |  λ = {lambd*1e3:.4f} mm  |  z = {z} m",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("06_rs_thz_result.png", dpi=150)
plt.show()
print("Saved: 06_rs_thz_result.png")
