"""Example 2: Phase-mask optimisation for a 1→4 beam splitter.

We optimise a phase-only SLM mask that splits one input Gaussian beam
into 4 equal-power output ports arranged on a circle.

Workflow:
  1. Build 4-port geometry
  2. Construct target intensity and port masks
  3. Initialise phase with phi_init_paper()
  4. Minimise ratio_loss with Adam (optax)
  5. Plot results
"""

import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from jaxoptics import (
    gauss_profile,
    make_transfer_func,
    propagate_asm,
    phaseshiftt,
    intensity,
    circle_centers_m,
    target_intensity_xy,
    make_port_masks,
    port_powers,
    normalize_ratios,
    ratio_loss,
    phi_init_paper,
    wrap_to_pi,
)

# ── Parameters ───────────────────────────────────────────────────────────────
num       = 2048
dx_m      = 0.58e-6
lambd_m   = 1550e-9
z_m       = 6e-3
rad_in    = 80e-6
rad_out   = 80e-6
n_ports   = 4
radius_m  = 430e-6
sigma_m   = 40e-6
P_target  = jnp.ones(n_ports, dtype=jnp.float32)   # equal split

# ── Geometry ─────────────────────────────────────────────────────────────────
amp_in, X, Y = gauss_profile(rad_in, num, dx_m)
H            = make_transfer_func(num, dx_m, z_m, lambd_m)
centers      = circle_centers_m(n_ports, radius_m)
masks        = make_port_masks(X, Y, centers, sigma_m)
I_tgt        = target_intensity_xy(X, Y, centers, rad_out, P_target)

# ── Anti-mask (penalise energy outside ports) ────────────────────────────────
no       = jnp.where(I_tgt > 0.19, 1.0, 0.0)
antimask = 1.0 - no

# ── Forward model ────────────────────────────────────────────────────────────
def forward(phi):
    E_in  = phaseshiftt(phi, amp_in)
    E_out = propagate_asm(E_in, H)
    return intensity(E_out)

@jax.jit
def loss_fn(phi):
    I_out      = forward(phi)
    P_out      = port_powers(I_out, masks)
    P_in_total = jnp.sum(intensity(amp_in))
    P_out_sum  = jnp.sum(P_out)
    return (
        ratio_loss(P_out, P_target)
        + 0.08 * jnp.abs(P_in_total - P_out_sum)
        + 3e-6 * jnp.sum(antimask * I_out)
    )

# ── Initialise phase ─────────────────────────────────────────────────────────
phi = phi_init_paper(X, Y, centers_m=centers, lambd_m=lambd_m, z_m=z_m)

# ── Optimisation loop ─────────────────────────────────────────────────────────
tx        = optax.adam(learning_rate=4e-1)
opt_state = tx.init(phi)

@jax.jit
def train_step(phi, opt_state):
    loss_val, grads  = jax.value_and_grad(loss_fn)(phi)
    updates, opt_state = tx.update(grads, opt_state, phi)
    phi = wrap_to_pi(optax.apply_updates(phi, updates))
    return phi, opt_state, loss_val

for step in range(201):
    phi, opt_state, L = train_step(phi, opt_state)
    if step % 20 == 0:
        P_out = port_powers(forward(phi), masks)
        print(f'step {step:4d}  loss={float(L):.4f}  ratios={jnp.array(normalize_ratios(P_out))}')

# ── Visualise results ────────────────────────────────────────────────────────
extent = [
    float(X.min()) * 1e6, float(X.max()) * 1e6,
    float(Y.min()) * 1e6, float(Y.max()) * 1e6,
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].imshow(intensity(amp_in), extent=extent, cmap='hot', origin='lower')
axes[0].set_title('Input intensity')

axes[1].imshow(forward(phi), extent=extent, cmap='hot', origin='lower')
axes[1].set_title('Output intensity (optimised)')

axes[2].imshow(phi, extent=extent, cmap='RdBu', origin='lower')
axes[2].set_title('Phase mask')

for ax in axes:
    ax.set_xlabel('x [µm]')
    ax.set_ylabel('y [µm]')
plt.tight_layout()
plt.savefig('02_beamsplitter.png', dpi=150)
plt.show()
print('Saved 02_beamsplitter.png')
