# Zoom Structure Generation

Source: `test_zoom/test_gen_zoom.py`

This example searches for zoom-lens group structures and motion trajectories before refining full optical prescriptions.

## Available configurations

The included YAML files cover nominal zoom ratios of 2×, 4×, 7×, 10×, 20×, and 40×.

```bash
cd test_zoom
python test_gen_zoom.py --name gen_zoom_2x
```

## Workflow

1. Read target focal lengths, F-numbers, fields, total track, back focus, group topology, and gap limits.
2. Create `lens.Zoom` and initialize a large batch of first-order candidates.
3. Run simulated annealing and gradient-based first-order refinement.
4. Convert selected candidates into multi-configuration physical systems.
5. Register averaged gradients for parameters shared by zoom groups.
6. Optimize curvature, spacing, material, and high-order surface coefficients in stages.
7. Apply zoom-specific merit terms for fixed groups, motion smoothness, and motion-class air gaps.
8. Preserve the best candidate data and save ranked prescriptions.

## Multi-configuration conventions

`CFG_NUM` is the number of zoom positions. `MAX_VIEW`, EFL targets, and F-number targets contain one value per configuration. Surface thickness can also be a list. `zoom_type` labels determine which spacings or groups remain fixed and which move. This consumes HappyLens's only configuration axis: independent operating conditions cannot be evaluated as an additional parallel dimension and must instead be simulated one at a time.

The physical-system builder generates two-character `zoom_type` values
(`FF`, `FM`, `MF`, `MM`) for individual surface entries. Later code uses
these values for cross-configuration gradient averaging and for `FIX_LENS`,
`SMOOTH_ZOOM`, and motion-class air-gap losses. They are implementation
metadata; preserve them when editing or reusing a generated zoom JSON rather
than trying to infer them only from whether a single surface appears fixed.

## Configuration parameter reference

The zoom YAML files share the runtime, spectral, merit, and staged-optimization
concepts used by prime generation, but all configuration-dependent targets
must align with `CFG_NUM`, and topology is nested by zoom group.

### Runtime, spectrum, and field sampling

| Key | Type / unit | Meaning |
| --- | --- | --- |
| `DEVICE` | string | PyTorch device. The workflow uses a large first-order population and fused Adam, so a CUDA device is expected by the supplied files. |
| `SEED` | list of integers | Performs one complete independent zoom search per seed. |
| `WAVELENGTHS` | list, mm | Traced wavelengths; `550.e-6` mm equals 550 nm. |
| `WAVEWEIGHTS` | list | Relative spectral weights; length must equal `WAVELENGTHS`. |
| `P_WAVE` | integer | Zero-based primary-wavelength index. |
| `CFG_NUM` | integer | Number of zoom positions/configurations. This controls the required length of `MAX_VIEW`, `MERIT.EFL.target`, and `MERIT.FNO.target`. |
| `MAX_VIEW` | list, degrees | Maximum half-field angle for every zoom position, in the same order as the focal-length targets. |
| `NORM_VIEWS` | list in `[0, 1]` | Normalized field samples evaluated at every zoom position. |
| `AZIMUTHS` | degrees | Field azimuth samples. The supplied rotationally symmetric searches use `[0.]`. |
| `VIG` | mapping or `null` | Optional normalized-pupil clipping with arrays `VUY`, `VLY`, `VUX`, and `VLX`. Each array aligns with `NORM_VIEWS`; `null` disables explicit clipping. |

For a five-position zoom, these lists align by index:

```yaml
CFG_NUM: 5
MAX_VIEW: [21.80, 4.57, 2.29, 1.53, 1.15]
MERIT:
  EFL: {target: [5., 25., 50., 75., 100.], weight: 50.}
  FNO: {target: [3., 3.63, 4.42, 5.21, 6.], weight: 1.}
```

Index `0` is the first zoom configuration and index `4` the last; the code
does not infer or sort wide/tele order.

### Zoom-group topology

```yaml
GROUP_STRUCTURE: 'FMMF'
GROUP_TYPE: 'SD|DS|DS|SDSD'
SURF_TYPE: 'SS|SS|SS|SSSS'
MAT_TYPE: 'RR|RR|RR|RRRR'
MAT_CATA: 'GG|GG|GG|GGGG'
STOP_POS: 2
STOP_FIX: true
```

| Key | Codes | Meaning |
| --- | --- | --- |
| `GROUP_STRUCTURE` | `F`, `M` | One character per optical group: `F` is fixed across zoom positions and `M` is movable. Its length is the number of groups. |
| `GROUP_TYPE` | groups separated by `\|`; `S`, `D` inside a group | Element makeup of each group. For example, `SD` is a singlet followed by a cemented doublet; `SDSD` contains four elements. |
| `SURF_TYPE` | `S`, `A`, `Q`, `q` | One surface-family code per element inside each group: spherical, even asphere, Q-con, or Q-bfs. Its nested shape must match `GROUP_TYPE`. |
| `MAT_TYPE` | `K`, `F`, `M`, `R` | One material-selection code per element. Singlets use crown (`K`), flint (`F`), or unrestricted (`R`); doublets use mixed crown/flint (`M`) or unrestricted (`R`). |
| `MAT_CATA` | `G`, `P` | One catalog code per element: glass or plastic. Its nested shape must also match `GROUP_TYPE`. |
| `STOP_POS` | integer | Number of groups before the inserted aperture stop. `2` inserts the stop after the first two groups and before group index 2. |
| `STOP_FIX` | boolean | Keeps the first-order stop behavior shared across zoom configurations and is forwarded to stop-radius updates during physical optimization. |

`GROUP_TYPE`, `SURF_TYPE`, `MAT_TYPE`, and `MAT_CATA` must contain exactly one
pipe-separated entry per character in `GROUP_STRUCTURE`; within each group,
their character counts must match.

### First-order zoom initialization and physical population

| Key | Type / unit | Meaning |
| --- | --- | --- |
| `DELANO_SYS` | integer | Number of first-order zoom candidates generated in parallel. |
| `LR_DELANO` | float | Learning rate for optimizing the first-order zoom trajectories. |
| `SYS_NUM` | integer | Number of full physical zoom prescriptions retained for later optimization. Higher zoom-ratio files lower this value to control memory. |
| `PRE_SAMP` | integer or `null` | Pre-sampling order used when establishing pupils and clear apertures. |
| `SAMP_METHOD` | string | Pupil distribution used for merit evaluation and saved analysis. Supported values are `square`, `hexapolar`, `fibonacci`, `ring`, and `line`; the provided files use `ring`. |
| `SAMP_RAYS` | integer | Sampling order passed to `Generation_Zoom`. |
| `MAX_RADIUS` | mm or `null` | Optional clear-aperture radius cap. |
| `FIX_RADIUS_SURF` | list | Reserved field; `test_gen_zoom.py` currently does not consume it. |

### Zoom merit operands

The common operand meanings match the
[prime reference](prime_generation.md#prime-lens-merit-operands). For zoom
systems, scalar bounds generally apply to all configurations, while `EFL` and
`FNO` targets are configuration lists.

| Operand | Parameters | Zoom-specific interpretation |
| --- | --- | --- |
| `EFL` | `target` list (mm), `weight` | Effective focal length at every zoom position. List length must equal `CFG_NUM`. |
| `FNO` | `target` list, `weight` | Per-position upper F-number targets; values above the target are penalized. |
| `SPOT` | `ref`, `k`, `weight` | Summed geometric spot penalty across wavelengths, fields, azimuths, and all zoom positions. |
| `DISTOR` | `target`, optional `abs`, `weight` | Fractional distortion bound applied across zoom positions. |
| `LATERAL` | `ref`, `weight` | Wavelength-dependent lateral displacement relative to `rms` or `chief`, normalized by the primary-wavelength Airy scale. |
| `BFL` | `target` (mm), `weight` | Minimum last-surface-to-image clearance for every configuration. |
| `TOTR` | `target` (mm), `weight` | Maximum total optical track. |
| `GLA_MIN_THICK`, `GLA_MAX_THICK` | `min_thick`/`max_thick`, `td_ratio`, `weight` | Glass-thickness constraints over all configurations.<br>In `MeritZ`, `td_ratio` takes precedence when both absolute and ratio fields are present. |
| `AIR_THICK` | `FF_target`, `FM_target`, `MM_target` (mm), `weight` | Minimum sag-aware air gaps for fixed-to-fixed, fixed-to-moving (either direction), and moving-to-moving boundaries. |
| `FIX_LENS` | `weight` | Penalizes configuration-dependent axial changes in spacings classified as fixed. |
| `SMOOTH_ZOOM` | `weight` | Encourages each moving spacing to vary monotonically through the listed configuration order. |
| `SURF_K` | `target` (degrees), `weight` | Surface-normal manufacturability limit. |
| `GLA_Z` | `z_min`, `weight` | Lower bound on the implemented glass-curvature separation metric. |
| `ANGLE` | `target` (degrees), `weight` | Maximum incidence/refraction angle. |
| `CRA` | `target` (degrees), `weight` | Optional image-space chief-ray-angle limit. |
| `ROC` | indexed `s_id`, `sign`, `weight` | Optional physical-surface curvature-sign constraint. Surface IDs refer to the expanded zoom prescription. |

`MeritZ.forward_loss()` does not dispatch `SURF_GAP`; use the motion-class
`AIR_THICK` fields for zoom air-gap control. Full mathematical behavior is
summarized under [Zoom merit](../api/optimization.md#zoom-merit-meritz).

### Staged and hybrid optimization

These fields behave as in prime generation:

| Key | Meaning |
| --- | --- |
| `EPOCH` | Adam iteration count for each stage. |
| `ADD_PARAM` | High-order parameters enabled at each stage (`ai4`, `ai6`, `conic`, `qi*`, or `null`). |
| `OPT_MAT_STAGE` | `1` optimizes continuous material coordinates; `0` fits and freezes catalog materials. |
| `LR_OPT`, `LR_SCALE` | Base learning rate and per-parameter scaling used by `Generation_Zoom.params_lr()`. |
| `DROPOUT` | Candidate-level gradient-drop probability. |
| `MAX_STATE_T` | Iterations used to ramp fields from 50% to 100% and tighten F-number targets to their configured values. |
| `OPTIC_UPDATE_FREQ` | Frequency of stop/aperture/focus maintenance and invalid-candidate replacement. |
| `SAVE_FREQ` | Frequency of intermediate analysis snapshots. |
| `P_FIX`, `TOP_PICK` | Fraction protected during element/material diversification, and whether protection selects the best or random candidates. |
| `HYBRID_MODE` | Stage-boundary population-search control: `switch` enters the configured search; `auto` can skip it based on stored-best/current losses after continuation. |
| `SHADE.iters`, `F`, `CR`, `c`, `p` | SHADE iteration count, initial differential/crossover values, adaptive-memory count, and p-best fraction. |
| `PRE_OPT` | Only activates the initial pre-optimization path when a separate `DE` block exists. The supplied zoom YAML files have no `DE` block, so this field currently has no effect. |

Optional `DE`, `SA`, and `GA` blocks follow the same schemas and execution
order documented in the [prime workflow](prime_generation.md#staged-local-and-population-optimization).
`DE` supplies the initial SHADE-style pass gated by `PRE_OPT`; stage-end
population methods run in the order SA, GA, then SHADE when their blocks are
present.

The three stage lists must be aligned:

```text
len(EPOCH) == len(ADD_PARAM) == len(OPT_MAT_STAGE)
```

### Consistency checklist

- `len(WAVELENGTHS) == len(WAVEWEIGHTS)` and `P_WAVE` is a valid zero-based index.
- `len(MAX_VIEW) == len(MERIT.EFL.target) == len(MERIT.FNO.target) == CFG_NUM`.
- `len(GROUP_STRUCTURE)` equals the number of pipe-separated entries in each nested topology string.
- Within every group, `GROUP_TYPE`, `SURF_TYPE`, `MAT_TYPE`, and `MAT_CATA` contain the same number of element codes.
- `len(EPOCH) == len(ADD_PARAM) == len(OPT_MAT_STAGE)`.
- Zoom configurations are interpreted in the order written; that order should follow a meaningful wide-to-tele or tele-to-wide trajectory for `SMOOTH_ZOOM`.

## Outputs

Results are stored below:

```text
test_zoom/results/<configuration-name>/<timestamp>/
```

Each seed receives its own child directory. Best systems are stored in `min_loss_*` directories with analysis figures and JSON prescriptions.

High zoom ratios represent much harder search problems. Start with the 2× configuration when validating a new environment or algorithm change.
