"""Example 3: Superposition of 4 tilted Gaussian beams as input.

Shows how to build a multi-beam input field from 4 plane-wave-tilted
Gaussian beams (±theta in x and y), propagate it, and visualise the result.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxoptics import gauss_profile, make_transfer_func, propagate_asm, intensity

# ── Parameters ───────────────────────────────────────────────────────────────
num     = 2048
dx_m    = 0.58e-6
lambd_m = 1550e-9
z_m     = 6e-3
rad_m   = 80e-6
theta   = 10.0          # tilt angle [degrees]

# ── Build tilted beam superposition ──────────────────────────────────────────
_, X, Y = gauss_profile(rad_m, num, dx_m)
k0   = 2 * jnp.pi / lambd_m
kx   = k0 * jnp.sin(jnp.deg2rad(theta))

amp = gauss_profile(rad_m, num, dx_m)[0]
E1  = amp * jnp.exp( 1j * kx * Y)   # +theta along x
E2  = amp * jnp.exp(-1j * kx * Y)   # -theta along x
E3  = amp * jnp.exp( 1j * kx * X)   # +theta along y
E4  = amp * jnp.exp(-1j * kx * X)   # -theta along y
E_in = E1 + E2 + E3 + E4

# ── Propagate ────────────────────────────────────────────────────────────────
H     = make_transfer_func(num, dx_m, z_m, lambd_m)
E_out = propagate_asm(E_in, H)

# ── Visualise ────────────────────────────────────────────────────────────────
extent = [
    float(X.min()) * 1e6, float(X.max()) * 1e6,
    float(Y.min()) * 1e6, float(Y.max()) * 1e6,
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, field, title in zip(axes, [E_in, E_out], ['Input (4 tilted beams)', f'Propagated  z={z_m*1e3:.0f} mm']):
    ax.imshow(intensity(field), extent=extent, cmap='hot', origin='lower')
    ax.set_title(title)
    ax.set_xlabel('x [µm]')
    ax.set_ylabel('y [µm]')
plt.tight_layout()
plt.savefig('03_tilted_beams.png', dpi=150)
plt.show()
print('Saved 03_tilted_beams.png')
