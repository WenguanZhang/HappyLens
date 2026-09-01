# System

`lens.system.System` owns an optical prescription and provides the main interface for sampling, tracing, editing, and saving it. It inherits `torch.nn.Module`, so all trainable surface parameters participate in standard PyTorch state and optimization APIs.

## Construction

```python
System(
    wavelengths,
    p_wvl,
    max_view,
    waveweights=None,
    sys_num=1,
    cfg_num=1,
    pre_samp=None,
    stop_max_samp_ang=45.0,
    samp_method="square",
    fix_radius_surf=[],
    norm_views=None,
    azimuths=None,
    vig=None,
    **source,
)
```

### Common arguments

| Argument | Type | Description |
| --- | --- | --- |
| `wavelengths` | `list[float]` | Wavelengths in millimeters. |
| `p_wvl` | `int` | Index of the primary wavelength. |
| `max_view` | `float | list[float]` | Maximum half-field angle in degrees. A scalar is repeated over configurations. |
| `waveweights` | `list[float] | None` | Spectral weights; defaults to equal weights. |
| `sys_num` | `int` | Number of candidate systems stored in parallel. |
| `cfg_num` | `int` | Length of the single configuration axis.<br>A prime/fixed-focal-length system can use it for parallel discrete operating conditions.<br>Zoom/focus structures consume this axis for structural states, so additional operating conditions must be simulated one at a time. |
| `pre_samp` | `int | None` | Optional reverse pre-sampling order for severe pupil aberration. |
| `stop_max_samp_ang` | `float` | Maximum stop angle used by pre-sampling, in degrees. |
| `samp_method` | `str` | Default pupil distribution: `square`, `hexapolar`, `fibonacci`, `ring`, or `line`. |
| `fix_radius_surf` | `list[int]` | Surfaces whose clear-aperture radii remain fixed. Pass an explicit list because the current default is mutable. |
| `norm_views` | `list[float] | None` | Normalized fields; defaults to `[0,.3,.5,.707,.85,1]`. |
| `azimuths` | `list[float] | None` | Azimuths in degrees; defaults to four orthogonal directions. |
| `vig` | `dict | None` | Optional vignetting dictionary. |

### Prescription sources

Exactly one source mode is normally supplied:

| Keyword | Value | Purpose |
| --- | --- | --- |
| `file` | JSON path | Load an existing prescription. |
| `delano` | `(Delano, surface_type, material_type, catalog, merit)` | Instantiate a Delano-generated prime lens design. |
| `zoom` | `(Zoom, surface_type, material_type, catalog, merit)` | Instantiate a generated zoom design. |
| `random` | `(structure, surface_type, material_type, catalog, stop_pos, merit)` | Generate random prime lens candidates. |
| `random_zoom` | `(structure, element_type, surface_type, material_type, catalog, stop_pos, merit)` | Generate random zoom candidates. |

## Core attributes

| Attribute | Shape/type | Meaning |
| --- | --- | --- |
| `system` | `nn.ModuleList` | Object plane, optical surfaces, and image plane. |
| `stop_id` | `int` | Aperture-stop index in the flattened surface list. |
| `valid` | `[sys]` bool | Candidate-system validity mask. |
| `EFFL` | `[sys,cfg]` | Effective focal length. |
| `FNO` | `[sys,cfg]` | F-number. |
| `ENPP`, `EXPP` | `[sys,cfg]` | Entrance- and exit-pupil positions. |
| `ENPD`, `EXPD` | `[sys,cfg]` | Entrance- and exit-pupil diameters. |
| `TOTR` | `[sys,cfg]` | Axial track from the first optical surface to the image-side reference. |
| `zoom_type` | `list` | Optional two-character movement/parameter-sharing labels (`FF`, `FM`, `MF`, `MM`) read from JSON and consumed by zoom-specific gradient sharing and merit terms. |

`update()` recomputes all paraxial attributes from the current prescription. Call it after manually changing curvature or thickness.

## Loading and saving

### `read_sys(file)`

Parses a HappyLens JSON prescription and returns `(module_list, stop_id, zoom_type)`. Surface dictionaries are dispatched by their `type` field. Material catalog selection is inferred from the material name.

Material lookup uses the active glass and plastic catalogs, with `VACUUM` and
`MIRROR` handled specially. An unrecognized material name does not currently
produce a dedicated validation error: catalog selection falls through and the
surface constructor subsequently fails. Verify material names (and the active
catalog choice) before loading external or reference prescriptions.

The parser assumes ordered `OBJECT`/surface/`IMAGE` entries and exactly one
stop. It has no schema-validation pass: an unknown surface `type` is skipped
at construction rather than rejected immediately, and a missing stop leaves
`stop_id` undefined. See [Configuration and File Formats](../configuration.md)
before authoring prescriptions by hand.

This is normally called through `System(..., file=path)`.

### `save_json(sys_id, save_path)`

Serializes one candidate system. Tensor values are converted to plain lists or scalars. For a multi-configuration design, per-configuration thicknesses and radii remain lists.

Current round-trip limitations:

- remove tolerance `PACKAGE` wrappers before saving. `save_json()` iterates
  the top-level list, has no `PACKAGE` serialization branch, and currently
  fails when it later accesses the package as though it were a surface; nested
  assembly ownership is not representable in this JSON format;
- `Binary2` inherits from `Asphere` and is currently written through the asphere branch, so its diffraction order and phase coefficients are not preserved by `save_json()`.

Use the original prescription or a separate experiment record when these fields must be retained.

## Sampling and propagation

### `sample_ray_2d(...)`

```python
ray = system.sample_ray_2d(
    sampling,
    norm_view=None,
    azimuth=None,
    wavelength=None,
    pre_samp=None,
    samp_method=None,
    vig=None,
)
```

Creates a `Ray` with shape `[wav,sys,cfg,ang,azi,M,3]` for position and direction. `norm_view`, `azimuth`, and `wavelength` accept a Python float, list, or `None`; the current implementation does not accept caller-provided tensors for these arguments.

For an infinite object, directions are parallel for a given field. For a finite object, ray directions connect the object point to sampled entrance-pupil points.

### `propagate(ray, radius_flag=True, record=False)`

Traces through every optical element. When `radius_flag=True`, aperture checks update `ray.valid`. When `record=False`, returns the final `Ray`. When `record=True`, returns `(ray, positions, directions)`, where the additional tensors stack states at successive surfaces.

In the current implementation, both modes still construct the full
per-surface position and direction stacks internally; `record=False` only
omits them from the return value. Do not treat it as a memory-saving mode when
choosing candidate and ray-batch sizes.

### `reverse_propagate(ray)`

Traces from image space back toward object space. The implementation is not intended for systems already wrapped in tolerance `PACKAGE` objects.

### `pre_samp_ray(views, wavelengths, pre_samp)`

Advanced reverse-tracing helper that estimates usable pupil bounds for systems with strong pupil aberration. Returns `[sys,cfg,ang,4]`, representing two bounds for each normalized pupil coordinate.

## Data extraction and restoration

| Method | Result |
| --- | --- |
| `extract_surfs()` | Flattened list of surfaces; recursively expands `PACKAGE` objects. |
| `extract_tols()` | Dictionary of nested decenter and tilt parameters. |
| `extract_opt_data()` | Detached copies of all trainable optical parameters. |
| `extract_all_sys_data()` | Detached copies of surface optical data, including fixed aperture radii and Q-type normalization radii. |
| `fit_opt_data(opt_data)` | Copies a previous `extract_opt_data()` result back into the system. |

These methods are useful for keeping the best candidate during stochastic optimization without copying the complete module graph.

## Parameter control

```python
system.freeze_sys_param(surf_id, "conic")
system.unfreeze_sys_param(surf_id, "roc")
system.avg_sys_para_grad(surf_id, "thick")
```

- `freeze_sys_param()` and `unfreeze_sys_param()` change `requires_grad` for one parameter or a supported parameter group.
- for an unknown parameter name, the current `freeze_sys_param()` and
  `unfreeze_sys_param()` implementations return a `Warning` object instead of
  raising it. Check names against `named_parameters()`; a misspelled name can
  otherwise leave the training state unchanged without stopping execution;
- `avg_sys_para_grad()` is intended for configuration-shaped parameters such as `thick`; it averages their last gradient dimension. Do not apply it to a one-dimensional per-candidate parameter, because that last dimension is then the candidate-system dimension.
- `material_fit(surf_id, method='M')` maps continuous material coordinates
  back to a discrete entry in that surface's catalog. `method='M'` uses the
  code's covariance-weighted distance and `method='E'` uses Euclidean
  distance; other strings are unsupported. The surface batch must obey the
  homogeneous-catalog/special-medium restriction described under
  [Surfaces](surfaces.md).

Surface indices refer to `extract_surfs()`: index 0 is `OBJECT`, and the final index is `IMAGE`.

## Structural edits

| Method | Effect |
| --- | --- |
| `del_surfs(del_id)` | Deletes a continuous `[start,end]` surface interval and merges axial distances. It operates only on an unpackaged system and refuses to delete the stop. |
| `convert_sph_to_asp(surf_id, surf_type, order)` | Replaces a direct `System.system` spherical entry with `Asphere`, `Qcon`, or `Qbfs`. |
| `add_IRCF(thick, dist, mat, sampling=3)` | Inserts a plane-parallel glass-catalog infrared-cut filter before the image plane and subtracts `thick + dist` from the preceding image-side spacing. |
| `tilt_decenter_elements(start_surf, end_surf, decenter, tilt)` | Inserts legacy `Coordinate`/`Dummy` modules around a continuous surface range. For hierarchical tolerancing, prefer `PACKAGE` through `ini_tol_sys()`. |

After structural edits, re-check `stop_id`, trainable parameters, and optimizer parameter groups. Existing optimizer objects do not automatically discover newly created `nn.Parameter` objects.
Call `System.update()` after `del_surfs()` or `convert_sph_to_asp()` before
using cached paraxial attributes; `add_IRCF()` already calls it internally.
`del_surfs()` adjusts `stop_id` but does not remove or realign `zoom_type`
entries, so it should not be applied directly to a zoom prescription without
also repairing that metadata.

`convert_sph_to_asp()` preserves the basic radius, material, curvature,
thickness, conic, and catalog fields, but the replacement currently resets
custom aperture mode/bounds and `fix_radius`; it is therefore unsafe to apply
unchanged to an annular/obscured or fixed-aperture surface. Q-type
normalization radius is initialized to `1.2 ×` the maximum clear radius.

For `add_IRCF()`, `mat` must exist in the glass catalog because the inserted
front surface is constructed with `mat_cata='G'`. Ensure the pre-existing
last spacing is at least `thick + dist` to avoid creating a negative remaining
spacing.

## Reporting

`print_sys_para()`, `print_sys_grad()`, `print_tol_para()`, `print_tol_grad()`, and `print_thick_tol_para()` write to stdout or an optional logger. All accept candidate/configuration indices where applicable.

## Internal generation methods

`delano_sys()`, `zoom_sys()`, `rand_sys()`, `rand_zoom_sys()`, `simulated_annealing()`, and `paraxial_loss()` implement constructor source modes. They are advanced research interfaces and may mutate or filter candidate batches.
