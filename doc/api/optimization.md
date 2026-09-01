# Optimization

## `Merit`

```python
merit = lens.Merit(system, samp_rays=6)
```

`Merit` evaluates optical objectives over every candidate system and configuration. Its loss operators are built from PyTorch operations so that gradient-based optimizers can update parameters owned by `system`; the same class also provides in-place population-based and stochastic search methods.

## Parameter groups

```python
groups = merit.params_lr(lr=4e-5, scale=10.0)
optimizer = torch.optim.Adam(groups)
```

`params_lr()` assigns different learning-rate scales to curvature, thickness, conic constants, material coordinates, and high-order coefficients. Recreate the optimizer after adding, deleting, or converting surfaces.

## Combined objective

```python
loss_per_system = merit.forward_loss(
    args,
    writer=None,
    count=None,
    path=None,
)
```

The method traces and caches optimization rays, evaluates operands selected by `args`, applies each operand's `weight`, and returns `[sys]`. `writer` and `count` enable TensorBoard logging. `path` requests a loss-composition chart.

Typical use:

```python
optimizer.zero_grad()
loss = merit.forward_loss(cfg.MERIT)
loss[system.valid].sum().backward()
optimizer.step()
merit.update_system(rmax=cfg.MAX_RADIUS)
```

## Merit operands

| YAML key | Method | Important parameters | Purpose |
| --- | --- | --- | --- |
| `SPOT` | `spot_loss()` | `ref`, `k` | Squared geometric spot error relative to the selected reference. |
| `LATERAL` | `lateral_loss()` | `ref` (`rms` or `chief`) | Chromatic displacement from the wavelength-averaged image position, normalized by the primary-wavelength Airy scale. |
| `WAVEFRONT` | `wavefront_loss()` | `mode` (`rms` or `tv`) | Exit-pupil wavefront penalty. |
| `EFL` | `efl_loss()` | `target` | Effective focal length. |
| `FNO` | `fno_loss()` | `target` | Penalizes values above (slower than) the target; faster values are not penalized. |
| `TOTR` | `totr_loss()` | `target` | One-sided upper bound on total optical track; shorter systems are not penalized. |
| `BFL` | `bfl_loss()` | `target` | Last-surface clearance to image. |
| `GLA_MIN_THICK` | `gla_min_thick_loss()` | `td_ratio`, `min_thick`, `ircf` | Minimum glass edge/center thickness. |
| `GLA_MAX_THICK` | `gla_max_thick_loss()` | `td_ratio`, `max_thick`, `ircf` | Maximum glass thickness. |
| `GLA_MAX_MIN_RATIO` | `gla_max_min_ratio_loss()` | `max_ratio`, `ircf` | Element thickness uniformity. |
| `SAG_DIA_MAX_RATIO` | `sag_dia_max_ratio_loss()` | `max_ratio`, `ircf` | Surface sag relative to diameter. |
| `AIR_THICK` | `air_thick_loss()` | `target` | Prevents adjacent surfaces from intersecting. |
| `SURF_GAP` | `surf_gap_loss()` | `s_pre`, `s_aft`, `target`, `mode` | Constrains the minimum sampled sag-aware separation between selected surfaces. |
| `GLA_Z` | `gla_z_loss()` | `z_min`, `ircf` | Glass curvature/diameter separation constraint. |
| `ANGLE` | `angle_loss()` | `target` | Incidence/refraction-angle limit. |
| `CRA` | `cra_loss()` | `target` | Sensor chief-ray-angle control. |
| `ANGLE_STD` | `angle_std_loss()` | none | Penalizes traced ray-direction dispersion across surfaces and pupil rays, then aggregates it over configurations, fields, and azimuths. |
| `SURF_K` | `surf_k_loss()` | `target` (degrees) | Penalizes sampled surface normals whose angle to the optical axis exceeds the target. |
| `DISTOR` | `distor_loss()` | `target`, `absolute` | Bounds chief-ray displacement relative to the paraxial f-tan image height. |
| `PUPIL` | `pupil_loss()` | `ref_point_n` | Measures how well exit-pupil samples cover reference points on the paraxial exit-pupil rim. |
| `ROC` | `roc_loss()` | `surf_id`, `sign` | Curvature-sign constraint. |

Only the keys listed above are dispatched by `Merit.forward_loss()`. Unknown
keys are not guaranteed to raise a validation error, so validate configuration
names before a long run.

Important operand details:

- `SPOT.ref` may be `rms` (polychromatic centroid), `chief`
  (primary-wavelength chief ray), or `ideal` (f-tan target built from `efl`).
  `k` exponentially emphasizes larger ray errors. Through `forward_loss()`, an
  `EFL` entry is currently required even when `ref` is not `ideal`, because its
  target is always passed to `spot_loss()`.
- `WAVEFRONT.rms` currently returns mean-square OPD in waves squared without a
  final square root. `WAVEFRONT.tv` computes peak-to-valley (`max-min`), not
  spatial total variation.
- `GLA_Z` lower-bounds `abs(D₁·c₁ - D₂·c₂) / 4` for the two faces of each
  glass region, where `D` is clear diameter and `c` is stored curvature. It is
  dimensionless and is not a physical thickness.
- `DISTOR` uses the primary-wavelength chief ray. With `absolute=True`, both
  distortion signs are allowed inside `±target`; otherwise the allowed
  interval runs from zero to the signed target.

`seidel_loss()` is also available as a direct experimental helper for
spherical systems, but it is not dispatched by `forward_loss()`.

## Cached ray propagation

`propagate_all_rays()` fills internal tensors for positions, directions, optical paths, chief-ray indices, and validity. Operand methods use these caches to avoid retracing the same bundles.

## System maintenance

| Method | Description |
| --- | --- |
| `update_system(rmax=None, avg_cfg=False, fit_material=True, update_radius=True, quick_focus=True)` | Runs the standard post-update maintenance sequence. |
| `update_stop_radius(target, stop_fix=True)` | Adjusts stop radius to approach a target F-number. |
| `update_radius(rmax=None, only_img=False)` | Updates clear apertures from traced footprints. |
| `quickfocus(avg_cfg=False)` | Adjusts the final spacing to restore focus. |
| `reborn_bad_system(args, optimizer=None)` | Reinitializes invalid candidates. |

## Parallel population-based and stochastic optimization

| Method | Role |
| --- | --- |
| `random_flip_elements(...)` | Flips selected lens elements. |
| `random_change_materials(...)` | Samples alternative catalog materials. |
| `random_perturb_roc_thick(...)` | Perturbs curvature and spacing. |
| `random_perturb_Qtype(...)` | Perturbs Q-type coefficients. |
| `random_perturb_asphere(...)` | Perturbs aspheric coefficients. |
| `genetic_system(...)` | Genetic search over batched systems. |
| `differential_evolution_system(...)` | Differential evolution. |
| `differential_evolution_system_shade(...)` | SHADE-style adaptive differential evolution. |
| `simulated_annealing_system(...)` | Simulated annealing. |

These methods operate on the candidate-system batch and modify the system in place. Depending on the method, they may also update optimizer state, TensorBoard output, or saved candidates. Candidate-level parallelism is controlled by `System.sys_num` and the active PyTorch device.

| Method | Main controls |
| --- | --- |
| `genetic_system(args,iters,elitism_rate=.1,mutation_rate=.1,mutation_strength=.1,...)` | Elite fraction, per-candidate mutation probability, and parameter-type-scaled mutation amplitude. |
| `differential_evolution_system(args,iters,F=.5,CR=.5,...)` | Differential weight and candidate-level crossover probability. |
| `differential_evolution_system_shade(args,iters,F=.5,CR=.5,c=10,p=.2,...)` | Initial adaptive values, positive memory-channel count `c`, and p-best fraction `p`. |
| `simulated_annealing_system(args,T=10,T_min=1,step=.001,alpha=.9,iter=20,ptresh=.5,...)` | Initial/final temperature and perturbation scale.<br>Temperature multiplier, trials per temperature, and parameter-perturbation probability. |

All four search methods operate only on parameters currently returned by
`System.extract_opt_data()`, so frozen parameters are not part of the search.
They require at least one valid candidate for ranking/selection. If both
`writer` and `count` are supplied they advance and return `count`; otherwise
their in-place system update is the result and the return value is `None`.

## Accurate differentiable imaging

The following methods provide efficient PSF simulation for optical evaluation and end-to-end optimization across optics, ISP operations, and post-processing networks. The diffraction-aware methods are distinct from the faster geometric ray-splat approximation.

### `psf_rs(...)`

Returns Rayleigh–Sommerfeld PSFs with shape `[wav,sys,cfg,ang,azi,H,W]`. `auto=True` uses ordinary autograd operations; `auto=False` uses `RayleighSommerfeldPsfOp`. With `chief_o=True`, it also returns PSF centers shaped `[2,sys,cfg,ang,azi]`.

### `psf_rs_err(..., zernike_err, ...)`

Adds a Zernike wavefront error before diffraction propagation. `zernike_err` maps Noll indices to coefficients in waves at the selected or primary wavelength.

### `psf_co(...)`

Returns coherent-accumulation PSFs with the same seven-dimensional layout.

### `psf_spot(...)`

Returns geometric ray-splat PSFs. It is faster but excludes diffraction phase.
The `auto` argument is accepted but is not consulted by the current
implementation. There is also a current multi-field limitation: while filling
each field/azimuth plane, the aperture-validity mask is taken from field `0`,
azimuth `0`. Use this method only when ray validity is identical across the
requested samples, or render one field/azimuth at a time; use `Analysis.psf_spot()`
for a detached single-field diagnostic.

### `psf_to_rgb(psfs, psfs_weight, show=True)`

Converts `[wav,sys,cfg,ang,azi,H,W]` to `[3,sys,cfg,ang,azi,H,W]` using a `[3,wav]` weight matrix.

`relative_illumination()` returns `[sys,cfg,ang,azi]`; `distortion()` returns
`[sys,cfg,ang,azi,2]`. With `wavelength=None`, both trace only the primary
wavelength. Their current explicit-wavelength validation uses bitwise `~` on
`isinstance(...)`, so supplying any non-`None` wavelength raises
`ValueError`, including a Python float. Leave `wavelength=None` in the current
version; selecting another wavelength requires correcting that validation in
source.

## Zoom merit: `MeritZ`

`MeritZ` extends `Merit` for zoom prescriptions represented as multiple configurations of one optical structure:

- `forward_loss()` adds `FIX_LENS` and `SMOOTH_ZOOM` and uses motion-class parameters for `AIR_THICK`;
- `air_thick_loss(FF_target, FM_target, MM_target)` applies different gap limits by motion class;
- `fix_lens_loss()` enforces fixed-group consistency;
- `smooth_zoom_loss()` regularizes motion across configurations.

`MeritZ.forward_loss()` otherwise supports the `Merit` operands listed above except `SURF_GAP`.

## Damped least squares: `DLS`

```python
dls = lens.DLS(args=cfg.MERIT, system=system, samp_rays=cfg.SAMP_RAYS)
dls.step()
```

`DLS` constructs a residual vector and an explicit Jacobian for all trainable system parameters.

| Method | Return/effect |
| --- | --- |
| `update_variables()` | Refreshes the tuple of trainable parameters. |
| `jacobian()` | Returns `[sys,cfg,M,N]`, with residual dimension `M` and variable dimension `N`. |
| `calc_loss()` | Returns `[M,sys]`. |
| `step()` | Applies one adaptive Levenberg–Marquardt-style update. |

`mu` stores per-system damping. `max_iter`, `min_damp`, and `max_damp` control the retry loop. `_loss()`, `_spot_loss_vector()`, and `_wavefront_loss_vector()` are internal residual builders.

`DLS._loss()` supports `EFL`, `FNO`, `SPOT`, `WAVEFRONT`, `DISTOR`, `BFL`, `TOTR`, `GLA_MIN_THICK`, `GLA_MAX_THICK`, `GLA_MAX_MIN_RATIO`, `SAG_DIA_MAX_RATIO`, `AIR_THICK`, `SURF_K`, `ANGLE`, `CRA`, `ANGLE_STD`, `PUPIL`, and `GLA_Z`. Other `Merit` operands are not currently included in the DLS residual dispatcher.

Two current DLS details matter when reusing a `MERIT` dictionary:

- its vector wavefront builder accepts uppercase `RMS` and `TV`, whereas
  `Merit.forward_loss()` accepts lowercase `rms` and `tv`;
- DLS `RMS` is also a mean-square residual (after subtracting the sampled OPD
  mean), not a square-root RMS value;
- `step()` changes parameter data but does not run the normal
  `Merit.update_system()` maintenance sequence. Refresh focus, apertures,
  material fitting, and paraxial attributes explicitly when the workflow
  requires them.
