"""Sanity checks for propagate_asm correctness.

Four independent tests:
  1. Gaussian beam divergence  — compare simulated w(z) with analytic formula
  2. Energy conservation       — total power must be preserved across all z
  3. Plane wave propagation    — flat-top amplitude must stay flat, only phase changes
  4. Thin lens focusing        — spherical phase mask must focus beam to expected spot

Run from the repo root:
    python examples/00_sanity_checks.py
"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from jaxoptics import (
    gauss_profile,
    make_transfer_func,
    propagate_asm,
    phaseshift,
    intensity,
)

# ── shared grid parameters ────────────────────────────────────────────────────
NUM     = 1024
DX_M    = 0.5e-6        # 0.5 µm pixel pitch
LAMBD   = 1550e-9       # 1550 nm
W0      = 50e-6         # input beam waist  [m]
ZR      = jnp.pi * W0**2 / LAMBD   # Rayleigh range  [m]

print(f"Rayleigh range  zR = {float(ZR)*1e3:.2f} mm")
print(f"Grid size          = {NUM}x{NUM},  dx = {DX_M*1e6:.2f} µm")
print(f"Field of view      = {NUM*DX_M*1e3:.3f} mm\n")

# ── helper: 1/e^2 intensity radius from a 2-D field ──────────────────────────
def beam_radius(E, X, Y):
    """Second-moment (D4σ/2) beam radius from complex field E."""
    I   = jnp.abs(E)**2
    P   = jnp.sum(I) + 1e-30
    x0  = jnp.sum(I * X) / P
    y0  = jnp.sum(I * Y) / P
    r2  = (X - x0)**2 + (Y - y0)**2
    return float(jnp.sqrt(jnp.sum(I * r2) / P))


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Gaussian beam divergence
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1: Gaussian beam divergence")
print("=" * 60)

amp, X, Y = gauss_profile(W0, NUM, DX_M)
E_in      = amp.astype(jnp.complex64)

z_values  = np.linspace(0.1, 4.0, 10) * float(ZR)   # 0.1 … 4 × zR
w_sim     = []
w_analytic = []

for z in z_values:
    H     = make_transfer_func(NUM, DX_M, float(z), LAMBD)
    E_out = propagate_asm(E_in, H)
    w_sim.append(beam_radius(E_out, X, Y))
    w_analytic.append(float(W0 * jnp.sqrt(1 + (z / ZR)**2)))

w_sim      = np.array(w_sim)      * 1e6   # → µm
w_analytic = np.array(w_analytic) * 1e6
err1 = np.abs(w_sim - w_analytic) / w_analytic * 100
print(f"  Max relative error vs analytic w(z): {err1.max():.2f} %")
assert err1.max() < 5.0, f"FAIL: divergence error {err1.max():.1f} % > 5 %"
print("  PASS")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Energy conservation
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 2: Energy conservation")
print("=" * 60)

P_in = float(jnp.sum(intensity(E_in)))
print(f"  Input power: {P_in:.6f}")

z_test    = [0.5, 1.0, 2.0, 5.0, 10.0]   # mm
pass_flag = True
for z_mm in z_test:
    H      = make_transfer_func(NUM, DX_M, z_mm * 1e-3, LAMBD)
    E_out  = propagate_asm(E_in, H)
    P_out  = float(jnp.sum(intensity(E_out)))
    rel    = abs(P_out - P_in) / P_in * 100
    status = "OK" if rel < 0.5 else "WARN"
    print(f"  z={z_mm:5.1f} mm   P_out={P_out:.6f}   err={rel:.3f} %   [{status}]")
    if rel >= 0.5:
        pass_flag = False

assert pass_flag, "FAIL: energy not conserved within 0.5 %"
print("  PASS")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Plane wave stays flat
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 3: Plane wave amplitude invariance")
print("=" * 60)

# Soft-aperture plane wave (Gaussian envelope, wide w = half the FOV)
W_pw     = NUM * DX_M / 4          # 128 µm — fills most of the grid
amp_pw, _, _ = gauss_profile(W_pw, NUM, DX_M)
E_pw     = amp_pw.astype(jnp.complex64)
A_in     = jnp.abs(E_pw)

z_pw = 2e-3   # 2 mm
H    = make_transfer_func(NUM, DX_M, z_pw, LAMBD)
E_pw_out = propagate_asm(E_pw, H)
A_out    = jnp.abs(E_pw_out)

# Compare amplitudes only in the central region (avoid edge artefacts)
crop = NUM // 4
A_in_c  = A_in [NUM//2 - crop : NUM//2 + crop, NUM//2 - crop : NUM//2 + crop]
A_out_c = A_out[NUM//2 - crop : NUM//2 + crop, NUM//2 - crop : NUM//2 + crop]
rms_diff = float(jnp.sqrt(jnp.mean((A_in_c - A_out_c)**2))) / float(jnp.max(A_in_c)) * 100
print(f"  Central-region amplitude RMS change: {rms_diff:.3f} %")
assert rms_diff < 1.0, f"FAIL: plane-wave amplitude changed by {rms_diff:.2f} %"
print("  PASS")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Thin lens focusing
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("TEST 4: Thin lens focusing")
print("=" * 60)

F_m   = 3e-3    # focal length 3 mm
amp_l, Xl, Yl = gauss_profile(W0, NUM, DX_M)

# Thin-lens phase  phi_lens = -pi/lambda * (x^2+y^2) / f
k0        = 2 * jnp.pi / LAMBD
phi_lens  = -0.5 * k0 / F_m * (Xl**2 + Yl**2)
E_lens    = phaseshift(phi_lens, amp_l.astype(jnp.complex64))

H_focus   = make_transfer_func(NUM, DX_M, F_m, LAMBD)
E_focus   = propagate_asm(E_lens, H_focus)

# The focused spot should be much smaller than the input beam
w_focused = beam_radius(E_focus, Xl, Yl)
w_diffraction_limit = float(LAMBD * F_m / (jnp.pi * W0))   # approx
print(f"  Input waist         : {W0*1e6:.1f} µm")
print(f"  Focused spot (sim)  : {w_focused*1e6:.2f} µm")
print(f"  Diffraction limit   : {w_diffraction_limit*1e6:.2f} µm")
ratio = w_focused / w_diffraction_limit
print(f"  Ratio sim/theory    : {ratio:.3f}  (expect ≈ 1)")
assert 0.5 < ratio < 2.0, f"FAIL: focused spot {ratio:.2f}x off from diffraction limit"
print("  PASS")


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

extent_full = [
    float(X.min())*1e6, float(X.max())*1e6,
    float(Y.min())*1e6, float(Y.max())*1e6,
]

# ── plot 1: divergence curve ──────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(z_values / float(ZR), w_analytic, 'k--', lw=2, label='Analytic $w(z)$')
ax1.plot(z_values / float(ZR), w_sim,      'o-',  lw=2, label='ASM simulation', color='royalblue')
ax1.set_xlabel('$z / z_R$')
ax1.set_ylabel('Beam radius [µm]')
ax1.set_title('Test 1 — Gaussian divergence')
ax1.legend()
ax1.grid(True, alpha=0.3)

# ── plot 2: energy vs z ───────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2:4])
z_arr  = np.array(z_test)
P_arr  = []
for z_mm in z_test:
    H_tmp  = make_transfer_func(NUM, DX_M, z_mm * 1e-3, LAMBD)
    P_arr.append(float(jnp.sum(intensity(propagate_asm(E_in, H_tmp)))) / P_in)
ax2.plot(z_arr, np.array(P_arr), 's-', color='seagreen', lw=2)
ax2.axhline(1.0, color='k', lw=1, ls='--')
ax2.set_ylim(0.95, 1.05)
ax2.set_xlabel('z [mm]')
ax2.set_ylabel('$P_{out} / P_{in}$')
ax2.set_title('Test 2 — Energy conservation')
ax2.grid(True, alpha=0.3)

# ── plot 3: plane wave amplitude in / out ─────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.imshow(np.array(A_in),  extent=extent_full, cmap='hot', origin='lower', vmin=0)
ax3.set_title('Test 3 — Plane wave\n(input amplitude)')
ax3.set_xlabel('x [µm]'); ax3.set_ylabel('y [µm]')

ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(np.array(A_out), extent=extent_full, cmap='hot', origin='lower', vmin=0)
ax4.set_title(f'Test 3 — Plane wave\n(output  z={z_pw*1e3:.0f} mm)')
ax4.set_xlabel('x [µm]')

# ── plot 4: focused spot ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
cr  = NUM // 2
zoom = 80   # µm
crop_px = int(zoom / (DX_M * 1e6))
I_focus_crop = np.array(intensity(E_focus))[
    cr - crop_px : cr + crop_px,
    cr - crop_px : cr + crop_px,
]
ext_zoom = [-zoom, zoom, -zoom, zoom]
ax5.imshow(I_focus_crop, extent=ext_zoom, cmap='hot', origin='lower')
ax5.set_title(f'Test 4 — Focused spot\n(f = {F_m*1e3:.0f} mm)')
ax5.set_xlabel('x [µm]'); ax5.set_ylabel('y [µm]')

# ── plot 5: lens phase mask ───────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 3])
ax6.imshow(np.array(phi_lens % (2*np.pi) - np.pi),
           extent=extent_full, cmap='RdBu', origin='lower')
ax6.set_title('Test 4 — Lens phase mask\n(wrapped to [-π, π])')
ax6.set_xlabel('x [µm]')

fig.suptitle('jaxoptics  ·  propagate_asm sanity checks', fontsize=14, fontweight='bold')
plt.savefig('00_sanity_checks.png', dpi=150, bbox_inches='tight')
plt.show()

print()
print("All 4 tests passed. Figure saved to 00_sanity_checks.png")
