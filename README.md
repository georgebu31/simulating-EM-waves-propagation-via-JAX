# simulating-EM-waves-propagation-via-JAX

A lightweight JAX library for simulating free-space electromagnetic wave propagation and optimising phase-only spatial light modulator (SLM) masks.

## Motivation

This project is an attempt to reproduce and extend the results from:

> **Tian, et al.**
"Metasurface‐based free‐space multi‐port beam splitter with arbitrary power ratio." Advanced Optical Materials 11.20, 2300664 (2023).

The key idea from the paper is to optimise a phase-only mask (e.g. on an SLM) such that a single input Gaussian beam is split into multiple output ports with target power ratios. The phase initialisation heuristic (`phi_init_paper`) directly follows the superposition-of-blazed-gratings approach described therein. All simulation and optimisation code in this repo is written from scratch in JAX to achieve similar results with full GPU acceleration and automatic differentiation.

## Features

- **Angular Spectrum Method (ASM)** — exact scalar diffraction propagation via FFT
- **Gaussian beam generation** — centred amplitude profiles with arbitrary waist and pixel pitch
- **Phase manipulation** — apply phase masks, wrap to [-π, π]
- **Multi-port power analysis** — Gaussian integration masks, power ratios, MSE loss
- **Phase initialisation** — smart initial guesses (Tian et al. 2023 heuristic and aperture-weighted superposition)
- **Gradient-based optimisation** — fully JIT-compiled loss + `optax` Adam ready to use

## Installation

```bash
pip install jax jaxlib optax matplotlib
git clone https://github.com/georgebu31/simulating-EM-waves-propagation-via-JAX
cd simulating-EM-waves-propagation-via-JAX
```

Then add the repo root to `PYTHONPATH`, or install as editable:

```bash
pip install -e .
```

## Library structure

```
jaxoptics/
  __init__.py       # public API
  beams.py          # gauss_profile
  propagation.py    # make_transfer_func, propagate_asm
  phase.py          # phaseshiftt, phaseshift, wrap_to_pi
  intensity.py      # intensity
  ports.py          # circle_centers_m, target_intensity_xy,
                    # make_port_masks, port_powers, normalize_ratios, ratio_loss
  init_phase.py     # make_phi_init, phi_init_paper
```

## Quick start

```python
import jax.numpy as jnp
from jaxoptics import gauss_profile, make_transfer_func, propagate_asm, phaseshiftt, intensity

num, dx_m, lambd_m, z_m = 1024, 0.58e-6, 1550e-9, 6e-3
amp, X, Y = gauss_profile(80e-6, num, dx_m)
H     = make_transfer_func(num, dx_m, z_m, lambd_m)
E_out = propagate_asm(phaseshiftt(jnp.zeros_like(amp), amp), H)
print("Output power:", float(jnp.sum(intensity(E_out))))
```

## Examples

| Script | Description |
|---|---|
| `examples/01_free_space_propagation.py` | Basic Gaussian beam propagation |
| `examples/02_beamsplitter_optimization.py` | Phase-mask optimisation for 1→4 beam splitter |
| `examples/03_tilted_beam_input.py` | Multi-beam (tilted) input field |

Run any example from the repo root:

```bash
python examples/01_free_space_propagation.py
```

## Physics background

The propagation kernel is the Angular Spectrum Method (ASM):

$$H(f_x, f_y) = \exp\!\left(i z \sqrt{k^2 - (2\pi f_x)^2 - (2\pi f_y)^2}\right)$$

where $k = 2\pi/\lambda$.  Evanescent waves ($k^2 < k_x^2 + k_y^2$) are automatically handled by the complex square root.
