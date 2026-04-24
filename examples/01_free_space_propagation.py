"""Example 1: Free-space propagation of a Gaussian beam using ASM.

Demonstrates basic usage of jaxoptics:
  - Build a Gaussian input field
  - Compute the ASM transfer function
  - Propagate and visualise intensity
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxoptics import gauss_profile, make_transfer_func, propagate_asm, phaseshiftt, intensity

# ── Simulation parameters ────────────────────────────────────────────────────
num     = 1024          # grid size (pixels)
dx_m    = 0.58e-6       # pixel pitch  [m]
lambd_m = 1550e-9       # wavelength   [m]
rad_m   = 80e-6         # beam waist   [m]
z_m     = 6e-3          # propagation  [m]

# ── Build input field ────────────────────────────────────────────────────────
amp, X, Y = gauss_profile(rad_m, num, dx_m)
phi       = jnp.zeros_like(amp)
E_in      = phaseshiftt(phi, amp)          # complex field: amp * exp(i*0)

# ── Propagate ────────────────────────────────────────────────────────────────
H     = make_transfer_func(num, dx_m, z_m, lambd_m)
E_out = propagate_asm(E_in, H)

# ── Visualise ────────────────────────────────────────────────────────────────
extent = [
    float(X.min()) * 1e6, float(X.max()) * 1e6,
    float(Y.min()) * 1e6, float(Y.max()) * 1e6,
]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, field, title in zip(axes, [E_in, E_out], [f'Input  z=0', f'Output z={z_m*1e3:.1f} mm']):
    ax.imshow(intensity(field), extent=extent, cmap='hot', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('x [µm]')
    ax.set_ylabel('y [µm]')
plt.tight_layout()
plt.savefig('01_propagation.png', dpi=150)
plt.show()
print('Saved 01_propagation.png')
