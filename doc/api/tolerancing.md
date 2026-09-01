# Tolerancing

HappyLens models decenter and tilt at the surface, element, or group level with nested `PACKAGE` modules, and models surface-spacing errors with `thick_tol` tensors. Any continuous surface range can be packaged, and packages may be nested to represent hierarchical assemblies.

## Tolerance dictionary

```python
tolerances = {
    "1_2": {
        "decenter": [0.0, 0.0],
        "tilt": [0.0, 0.0, 0.0],
    }
}
system.ini_tol_sys(tolerances)
```

A key such as `1_2` identifies a continuous surface range. The selected surface, element, or group is wrapped in a `PACKAGE`. Repeated calls can create nested packages for hierarchical assembly errors.

The indices are interpreted against the current top-level `System.system`
list at the start of each call, with `0` reserved for `OBJECT`. On the first
call for a flat prescription they therefore match physical surface IDs. After
packaging, a top-level entry may itself be a `PACKAGE`; a later call can wrap
those package entries to create another hierarchy level. Within one dictionary,
the implementation compensates for earlier ranges being collapsed, so provide
non-overlapping ranges in increasing order. It does not validate overlaps or
ordering.

## System tolerance methods

| Method | Description |
| --- | --- |
| `ini_tol_sys(tols_dic)` | Creates package transforms from a dictionary. |
| `remove_tol_param()` | Zeros decenter, tilt, and thickness perturbations, removes current packaging, and restores a flat system list. |
| `extract_tols()` | Returns a dictionary of transform parameter tensors, including nested paths. |
| `freeze_tol_param(tols_id,param_name)` | Disables gradients for a decenter or tilt parameter. |
| `unfreeze_tol_param(tols_id,param_name)` | Enables gradients. |
| `set_tol_param(tols_id,decenter=None,tilt=None)` | Sets deterministic errors. |
| `rand_tol_param(decenter_scale,tilt_scale)` | Samples random rigid-body errors. |
| `rand_decenter_tilt_thick_param(decenter_scale,tilt_scale,thick_scale)` | Samples rigid-body and thickness errors together. |

`tols_id` follows the path returned by `extract_tols()`, not necessarily the original range key after packages are nested.

For `set_tol_param()`, pass one `torch.Tensor` decenter vector `[dx,dy]` in
millimeters and/or one `torch.Tensor` tilt vector `[tx,ty,tz]` in degrees;
plain Python lists are not accepted by this setter. The current implementation broadcasts
that vector to every candidate and configuration, converts tilt to radians,
and stores both inputs with the internal inverse-transform sign. Use
`rand_tol_param()` when each candidate/configuration should receive an
independent sample; its `tilt_scale` is also expressed in degrees.

## Coordinate transformation

`Coordinate` and `PACKAGE` transform rays between the parent coordinate frame and the perturbed element frame. Decenter has two components. Tilt has three components and is converted into the quaternion/rotation representation used by the propagation code.

Because packages recursively contain surfaces, use `System.extract_surfs()` when a calculation needs the physical surface sequence and `System.system` when it needs assembly ownership.

## Thickness tolerance

Each `Sphere` stores `thick_tol` with the same `[sys,cfg]` shape as nominal thickness. `thickness()` is the value used during propagation. `update_thickness_tol(scale)` samples a zero-mean Gaussian error.

## Differentiable compensation

A typical nested-tolerance compensation loop is:

```python
system.rand_decenter_tilt_thick_param(
    decenter_scale,
    tilt_scale,
    thickness_scale,
)
merit.update_system(update_radius=False, quick_focus=True)

psfs, centers = merit.psf_rs(
    pupil_samp,
    image_samp,
    image_delta,
    fields,
    azimuths,
    auto=False,
    chief_o=True,
)
```

The rendered PSFs and centers can condition a restoration network. Freeze optical or tolerance parameters when training only digital compensation, and explicitly control dtype when moving from double-precision optics to single-precision networks.

## Reporting

- `print_tol_para()` and `print_tol_grad()` report rigid-body parameters.
- `print_thick_tol_para()` reports sampled thickness errors.
- `print_sys_para()` still reports nominal surface data; use `thickness()` when the perturbed physical spacing is required.
