"""Example 4: Zero-padded propagation — larger output FOV.

Demonstrates propagate_asm_padded vs propagate_asm for a diverging beam:
  - Input: 512×512 grid, dx=0.5µm  (FOV = 256µm)
  - Output options:
      pad_factor=1  → 512×512   (FOV = 256µm)  — beam clips at edge
      pad_factor=2  → 1024×1024 (FOV = 512µm)  — beam fits comfortably
      pad_factor=4  → 2048×2048 (FOV = 1024µm) — wide margin

Also runs nyquist_check() to show when padding becomes necessary.
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxoptics import (
    gauss_profile,
    propagate_asm,
    propagate_asm_padded,
    make_transfer_func,
    phaseshiftt,
    intensity,
    nyquist_check,
)

# ── parameters ─────────────────────────────────────────────────────────────
NUM     = 512
DX_M    = 0.5e-6
LAMBD   = 1550e-9
W0      = 30e-6
Z_M     = 8e-3          # 8 mm — enough for significant divergence

# ── Nyquist check for each pad_factor ──────────────────────────────────────
print("=" * 55)
print(f"Propagation distance z = {Z_M*1e3:.0f} mm")
for pf in [1, 2, 4]:
    print(f"\n--- pad_factor = {pf} ---")
    nyquist_check(NUM, DX_M, Z_M, LAMBD, pad_factor=pf)

# ── build input field ───────────────────────────────────────────────────────
amp, X, Y = gauss_profile(W0, NUM, DX_M)
E_in      = phaseshiftt(jnp.zeros_like(amp), amp)

# ── propagate with different pad factors ────────────────────────────────────
pad_factors = [1, 2, 4]
results     = {}
for pf in pad_factors:
    if pf == 1:
        H = make_transfer_func(NUM, DX_M, Z_M, LAMBD)
        results[pf] = intensity(propagate_asm(E_in, H))
    else:
        results[pf] = intensity(propagate_asm_padded(E_in, pf, DX_M, Z_M, LAMBD))

# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, pf in zip(axes, pad_factors):
    I    = results[pf]
    N_out = NUM * pf
    fov   = N_out * DX_M * 1e6           # µm
    ext   = [-fov/2, fov/2, -fov/2, fov/2]
    ax.imshow(I, extent=ext, cmap='hot', origin='lower')
    ax.set_title(f'pad_factor = {pf}\n{N_out}×{N_out}  FOV={fov:.0f} µm', fontsize=11)
    ax.set_xlabel('x [µm]')
    ax.set_ylabel('y [µm]')

fig.suptitle(
    f'Zero-padded ASM propagation   z={Z_M*1e3:.0f}mm   W₀={W0*1e6:.0f}µm   λ=1550nm',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('04_padded_propagation.png', dpi=150)
plt.show()
print('\nSaved 04_padded_propagation.png')

# ── power check: padding must conserve energy ───────────────────────────────
print("\nEnergy conservation check:")
P_in = float(jnp.sum(intensity(E_in)))
for pf, I in results.items():
    print(f"  pad={pf}  P_out/P_in = {float(jnp.sum(I))/P_in:.6f}")
