# Architecture Overview

HappyLens uses PyTorch as a shared computation backend. The optimization path can preserve gradients from optical parameters through ray propagation and diffraction-aware PSF formation, and onward through image degradation and neural restoration when the differentiable interfaces are used.

```text
JSON / YAML
    │
    ▼
 System ── contains ──► Surface modules
    │                       │
    │                       └─ Sphere / Asphere / Q-type / Binary2
    │
    ├─ sample & trace ──► Ray
    │
    ├─ report ──────────► Analysis
    │
    ├─ optimize ────────► Merit / MeritZ / DLS
    │
    ├─ search/edit ─────► Delano / Zoom / Generation / Deletion
    │
    └─ render PSF ──────► nets.simulate_rgb / simulate_raw
                              │
                              ▼
                         Restoration networks
```

## Optical core

`System` is the aggregate root. Its `nn.ModuleList` contains the object plane, optical surfaces, and image plane. Curvature, thickness, conic constants, aspheric coefficients, and continuous glass coordinates can all be represented by `nn.Parameter` objects.

`Ray` is a mutable data container rather than an `nn.Module`. It stores position, direction, optical path length, chief-ray indices, and validity masks. `System.propagate()` applies intersection, refraction, and coordinate updates surface by surface.

## Analysis and optimization

`Analysis` provides mostly `no_grad` visualization and reporting operations. `Merit` traces ray batches, caches intermediate data, and produces one loss per candidate system for standard PyTorch optimizers. `DLS` builds an explicit Jacobian on top of the same merit-function infrastructure.

## Computational imaging

`Merit.psf_rs()` and `Merit.psf_co()` preserve gradients with respect to optical parameters. Their PSFs can be passed directly to `nets.simulate_rgb()` or `nets.simulate_raw()` and then processed by a restoration network wrapped in `nets.Model`.

## Multiple systems and configurations

`sys_num` represents candidate designs evaluated in parallel. `cfg_num` is the
framework's only configuration axis. A prime/fixed-focal-length system may use
it for parallel discrete operating conditions; a zoom or focusing system uses
it for its structural positions. Those uses cannot be combined into two
independent dimensions, so a zoom/focus structure can evaluate only one
additional operating condition per run. Tolerance perturbations are parameters
applied to the existing `[sys,cfg]` states, not a separate configuration axis.
Most core tensors preserve both dimensions. `system.valid` marks only the
candidate-system dimension, so optimization commonly uses:

```python
loss[system.valid].sum().backward()
```
