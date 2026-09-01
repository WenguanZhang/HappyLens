# Structure Search

HappyLens uses first-order optimization to explore lens structures before full-prescription optimization. `Delano` and `Zoom` work mainly with paraxial variables; `System` converts selected candidates into physical surface modules; `Generation_Prime` and `Generation_Zoom` then refine the prescription. Both stochastic and gradient-based methods are available in this workflow.

The generation paths construct candidates from structure descriptors and sampled variables; they do not require a patent or lens library as an initialization source.

## Prime lens generation with `Delano`

```python
Delano(
    structure,
    sys_num,
    target_fov,
    target_effl,
    target_fno,
    target_totr,
    target_bfl,
    stop_pos,
    mat_type,
    dist_min=None,
)
```

`structure` and `mat_type` use `|`-separated element descriptors. A batch of candidate paraxial systems is normalized by target focal length.

### First-order state

- `y` and `ybar`: marginal- and chief-ray heights at group boundaries;
- `u` and `ubar`: corresponding paraxial slopes;
- `Q`: Lagrange invariant;
- group powers, separations, radii, and Seidel contributions derived by `generate()`.

### Main methods

| Method | Description |
| --- | --- |
| `linv()` | Computes the optical invariant. |
| `y_ybar_init()` | Samples initial paraxial ray heights. |
| `update_y_ybar()` | Refreshes constrained height tensors from trainable values. |
| `generate()` | Derives first-order lens quantities. |
| `calc_u_ubar()` | Computes paraxial slopes. |
| `calc_seidel(opt_y_ybar=True)` | Computes Seidel aberration contributions. |
| `calc_stop()` | Computes stop-related quantities. |
| `effl()` | Returns effective focal length. |
| `propagate(y0,u0)` | Performs paraxial propagation. |
| `fitness()` | Combines first-order objectives. |
| `optimize_SA()` | Runs simulated annealing. |
| `optimize(lr,save_dir)` | Runs gradient optimization and optionally saves figures. |
| `lens_instance(sys_num,cfg_num,stype,mat_type,mat_cata)` | Creates physical surfaces and a stop index for `System`. |

Individual terms include distance, stop, total-track, focal, angle, Seidel, and maximum-radius merits.

## Zoom generation with `Zoom`

```python
Zoom(
    group_structure,
    group_type,
    sys_num,
    target_fov,
    target_effl,
    target_fno,
    target_totr,
    target_bfl,
    FF_min_dist,
    FM_min_dist,
    MM_min_dist,
    stop_pos,
    stop_fix=True,
)
```

`group_structure` uses `F` for fixed and `M` for movable groups. `group_type` describes the lens structure inside each group. Target lists define one value per configuration.

| Method | Description |
| --- | --- |
| `linv()` | Computes the invariant for each configuration. |
| `y_ybar_init()` / `update_y_ybar()` | Initializes and constrains first-order ray heights. |
| `generate()` | Derives group powers and separations. |
| `effl()` | Computes focal length over the zoom range. |
| `fitness_A()` / `fitness_B()` | Two-stage structure objectives. |
| `optimize_SA()` | Runs stochastic first-order search. |
| `optimize(lr,save_dir)` | Runs gradient refinement. |
| `revise_lens_data()` | Repairs or normalizes generated lens data. |
| `lens_instance(...)` | Returns surfaces, stop index, and per-surface zoom labels. |

Zoom merit terms cover spacing, cross-configuration spacing variation, trajectory smoothness, radius/power variation, F-number, ray angles, and total track.

## Full-prescription generation classes

`Generation_Prime` extends `Merit`; `Generation_Zoom` extends `MeritZ`. Their principal override is `params_lr(lr, scale=10.)`, which assigns generation-specific learning-rate scales to curvature, thickness, conic constants, continuous material coordinates, even-aspheric coefficients, and Q-type coefficients.

## Component deletion with `Deletion`

`Deletion` extends `Merit` for continuously collapsing a selected lens element
before physically removing its surface interval.

```python
deletion = lens.Deletion(system=system, samp_rays=6)
del_id = deletion.find_del_surfs()
loss_opt = deletion.forward_loss(del_id, merit_args)
gap_residual, sag_residual = deletion.del_surf_loss(del_id)
```

| Method | Description |
| --- | --- |
| `find_del_surfs()` | Groups consecutive glass surfaces into elements and scores their power/aberration contribution.<br>Excludes the stop-containing element and returns `[start,end]`. |
| `del_surf_loss(del_id)` | Returns two `[sys]` residuals: the maximum internal sag-aware separation and maximum absolute sag over the selected interval. Driving both toward zero flattens and collapses the component. |
| `forward_loss(del_id,args,...)` | Evaluates the deletion-aware subset of merit operands. BFL and thickness/gap constraints account for the interval that is intended to disappear. |
| `params_lr(lr)` | Builds deletion-specific curvature, thickness, and conic parameter groups. |

`Deletion.forward_loss()` dispatches `EFL`, `FNO`, `SPOT`, `DISTOR`, `BFL`,
`TOTR`, `GLA_MIN_THICK`, `GLA_MAX_THICK`, `AIR_THICK`, and `SURF_K`. It does
not add `del_surf_loss()` automatically; the application loop chooses how to
weight and combine the two deletion residuals with the optical loss. Once the
collapse criterion is satisfied, call `System.del_surfs(del_id)` to perform
the structural edit and then rebuild any optimizer that should train the
remaining prescription.
