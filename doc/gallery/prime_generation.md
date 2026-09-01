# Prime Lens Structure Generation

Source: `test_prime/test_gen_prime.py`

This example generates prime lens prescriptions without requiring a patent or lens library as an initialization source. It supports a Delano-based first-order search path and a random physical-prescription initialization path.

## Available configurations

The repository currently includes:

- `gen_efl36_f3.5.yaml`;
- `gen_efl100_f2.8.yaml`;
- `gen_v33_f2.55.yaml`.

Run one configuration from `test_prime`:

```bash
cd test_prime
python test_gen_prime.py --name gen_efl36_f3.5
```

## Workflow

1. Create a timestamped result root and copy the YAML file.
2. Repeat the complete run for every value in `SEED`.
3. If the YAML contains a `DELANO` block, construct `lens.Delano`, run first-order simulated annealing and gradient-based refinement, then instantiate a physical `System`.
4. Otherwise create a batch of random systems from the structure descriptor.
5. Build a field-dependent vignetting model.
6. Freeze all parameters, then enable curvature, thickness, and selected high-order terms by optimization stage.
7. Alternate material fitting, continuous material optimization, local gradient updates, and candidate-system maintenance.
8. Keep detached copies of the best prescription data.
9. Rank final candidates and save JSON, layout, and spot-diagram results.

## Important configuration groups

| Group | Role |
| --- | --- |
| `STRUCTURE`, `SURF_TYPE`, `MAT_TYPE`, `MAT_CATA` | Defines element topology, surface family, material family, and catalog. |
| `DELANO` | Enables and configures first-order Delano search. |
| `MERIT` | Optical and manufacturability objectives. |
| `ADD_PARAM` | Adds trainable terms by optimization stage. |
| `OPT_MAT_STAGE` | Switches between fitted and continuous material optimization. |
| `EPOCH`, `LR_OPT`, `SAVE_FREQ` | Controls the main optimization loop. |

## Configuration parameter reference

The following reference describes the top-level keys used by
`test_prime/lens_yaml/*.yaml`. Lengths, units, and aligned topology strings are
significant; `GetYaml` does not perform schema validation before the search
starts.

### Runtime, spectrum, and field sampling

| Key | Type / unit | Meaning |
| --- | --- | --- |
| `DEVICE` | string | PyTorch device used for all tensors, normally `cuda:0`. The script uses fused Adam, so the selected device/backend must support it. |
| `SEED` | list of integers | Runs the complete generation workflow once per seed. More entries improve search diversity but multiply runtime and output size. |
| `WAVELENGTHS` | list, mm | Traced wavelengths. For example, `550.e-6` mm is 550 nm. |
| `WAVEWEIGHTS` | list | Relative merit weight for each wavelength. Its length must equal `WAVELENGTHS`; values need not sum to one. |
| `P_WAVE` | integer | Zero-based index of the primary wavelength used for paraxial quantities, chief-ray references, and several merit operands. |
| `MAX_VIEW` | degrees | Target maximum half-field angle. During continuation it starts at half this value and increases to the full value. |
| `NORM_VIEWS` | list in `[0, 1]` | Normalized field samples. Actual field angles equal `NORM_VIEWS × current MAX_VIEW`. Include `0` and `1` when both on-axis and full-field behavior matter. |
| `MAX_VIG` | normalized pupil fraction | Builds an upper-Y pupil clip that increases from zero on axis to this value at maximum field.<br>`0` disables it; the other three clipping directions remain zero. |
| `AZIMUTHS` | degrees | Field azimuths. `[0.]` evaluates the meridional direction used by the provided rotationally symmetric examples. |

### Lens topology

The four pipe-separated topology strings must contain the same number of
entries. Each entry describes one lens element rather than one optical
surface.

```yaml
STRUCTURE: 'S|D|D|S|S'
SURF_TYPE: 'S|S|S|S|S'
MAT_TYPE:  'K|M|M|R|K'
MAT_CATA:  'G|G|G|G|G'
STOP_POS: 2
```

| Key | Codes | Meaning |
| --- | --- | --- |
| `STRUCTURE` | `S`, `D` | `S` creates a singlet with two refracting surfaces; `D` creates a cemented doublet with three refracting surfaces. |
| `SURF_TYPE` | `S`, `A`, `Q`, `q` | Surface family applied to the element: spherical, even asphere, Q-con, or Q-bfs. High-order coefficients start at zero and are enabled through `ADD_PARAM`. |
| `MAT_TYPE` | `K`, `F`, `M`, `R` | Material-selection class.<br>`K`: crown-like; `F`: flint-like; `M`: crown/flint doublet; `R`: unrestricted.<br>Singlets use `K`/`F`/`R`; doublets use `M`/`R`. |
| `MAT_CATA` | `G`, `P` | Material catalog: optical glass (`G`) or plastic (`P`). |
| `STOP_POS` | integer | Number of lens elements before the inserted stop.<br>`0`: before the first element; `2`: after the second.<br>Later surface IDs include this stop and expanded singlet/doublet surfaces. |

All four topology strings must align entry by entry. A `D` still consumes only
one entry in each string even though it expands into three physical surfaces.

### First-order initialization and candidate population

| Key | Type / unit | Meaning |
| --- | --- | --- |
| `DELANO.NUM` | integer | Number of first-order Delano candidates. This is usually much larger than the physical candidate batch. |
| `DELANO.LR` | float | Learning rate for the Delano first-order optimization. |
| `SYS_NUM` | integer | Number of full physical prescriptions retained and optimized in parallel after first-order selection. GPU memory grows approximately with this population. |
| `CFG_NUM` | integer | Number of optical configurations. It must be `1` for these prime-generation workflows. |
| `PRE_SAMP` | integer or `null` | Pre-sampling order used while establishing pupils/clear apertures for difficult systems. Larger values are more robust but increase initialization cost. |
| `SAMP_METHOD` | string | Pupil-ray distribution used by the system and saved analysis. Supported values are `square`, `hexapolar`, `fibonacci`, `ring`, and `line`; the provided generation files use `ring`. |
| `SAMP_RAYS` | integer | Sampling order passed to `Generation_Prime` merit evaluation. Increase it for more reliable geometric metrics at higher cost. |
| `MAX_RADIUS` | mm or `null` | Optional upper bound applied when clear-aperture radii are updated. `null` leaves the update uncapped. |
| `FIX_RADIUS_SURF` | list of surface IDs | Reserved configuration field. `test_gen_prime.py` currently does not consume it, so listing IDs here does not freeze radii in this workflow. |

### Prime-lens merit operands

Every `MERIT` entry contributes its computed penalty multiplied by `weight`.
Targets are not all equality targets: several operands implement only an upper
or lower bound.

| Operand | Parameters | Interpretation |
| --- | --- | --- |
| `EFL` | `target` (mm), `weight` | Squared effective-focal-length error at the primary wavelength. |
| `FNO` | `target`, `weight` | Penalizes F-number values larger (slower) than the target; smaller values are not penalized by this operand. |
| `SPOT` | `ref`, `k`, `weight` | Geometric spot penalty. `ref` may be `rms`, `chief`, or `ideal`; `k` controls the exponential emphasis on rays farther from the reference. |
| `DISTOR` | `target`, optional `abs`, `weight` | Bounds f-tan(theta) distortion as a fraction; `0.01` means 1%. With the default `abs: true`, both signs are bounded. |
| `BFL` | `target` (mm), `weight` | Minimum back clearance from the last optical surface sag to the image plane. |
| `TOTR` | `target` (mm), `weight` | Maximum total optical track; only excess length is penalized. |
| `GLA_MIN_THICK` | `min_thick` (mm), `td_ratio`, `weight` | Minimum glass thickness. `td_ratio` specifies a diameter-relative lower bound; if both fields are present, the current prime merit adds both penalties. |
| `GLA_MAX_THICK` | `max_thick` (mm), `td_ratio`, `weight` | Maximum glass thickness using absolute and/or diameter-relative upper bounds. |
| `AIR_THICK` | `target` (mm), `weight` | Minimum air gap after surface sag is considered. |
| `SURF_K` | `target` (degrees), `weight` | Limits surface-normal tilt relative to the optical axis, acting as a manufacturability constraint. |
| `GLA_Z` | `z_min`, `weight` | Lower bound on the implemented dimensionless glass-curvature separation metric. |
| `CRA` | `target` (degrees), `weight` | Maximum image-space chief-ray angle. |
| `ANGLE` | `target` (degrees), `weight` | Maximum incidence/refraction angle across traced rays and surfaces. |
| `SURF_GAP` | indexed `s_pre`, `s_aft`, `target`, `mode`, `weight` | Constrains the sag-aware axial gap between two expanded surface IDs. `mode` is `gt`, `lt`, or `eq`. |
| `ROC` | indexed `s_id`, `sign`, `weight` | Optional curvature-sign constraint; `sign` is `p` for positive or `n` for negative curvature. |

See [Optimization](../api/optimization.md#merit-operands) for the complete
operand dispatcher. Removing an operand disables that objective; setting a
small weight only weakens it.

### Staged local and population optimization

`EPOCH`, `ADD_PARAM`, and `OPT_MAT_STAGE` describe the same sequence of
optimization stages and therefore must have identical lengths.

| Key | Meaning |
| --- | --- |
| `EPOCH` | Adam iterations in each stage. Total local-gradient iterations equal the sum of this list, excluding additional SHADE work. |
| `ADD_PARAM` | Parameters newly unfrozen at the start of each stage, or `null`. Examples include `conic`, `ai4`, `ai6`, and `qi0`...`qiN`. Once enabled, a parameter remains enabled in later stages. |
| `OPT_MAT_STAGE` | `1` keeps continuous material coordinates `g1/g2` trainable; `0` fits materials to the selected catalog and freezes those coordinates. Provided files normally use `0` in the final stage. |
| `LR_OPT` | Base Adam learning rate. |
| `LR_SCALE` | Parameter-type multiplier.<br>Approximately `LR_OPT/LR_SCALE` for curvature, `LR_OPT×LR_SCALE` for thickness, and `LR_OPT` for conic/Q terms; high-order aspheres use progressively smaller rates. |
| `DROPOUT` | Probability of randomly zeroing candidate-level gradients before each Adam step. This is gradient dropout, not neural-network dropout. |
| `MAX_STATE_T` | Continuation length. Over this many local iterations, field grows from 50% to 100% of `MAX_VIEW`, while the temporary F-number target tightens from twice the configured value to the final target. |
| `OPTIC_UPDATE_FREQ` | Frequency, in local iterations, for stop-radius adjustment, clear-aperture/focus maintenance, and invalid-candidate rebirth. |
| `SAVE_FREQ` | Frequency, in local iterations, for an `epoch_*` snapshot and analysis output. |

Population diversification is controlled by:

| Key | Meaning |
| --- | --- |
| `P_FIX` | Fraction of candidates protected from random element flips and material changes at stage transitions. |
| `TOP_PICK` | If `true`, protect the lowest-loss fraction; if `false`, protect a random fraction. |
| `HYBRID_MODE` | `switch` always enters configured stage-boundary population search; `auto` may skip it after `MAX_STATE_T` according to the stored-best/current-loss comparison. |
| `SHADE.iters` | SHADE iterations at each stage boundary reached by the script. |
| `SHADE.F`, `SHADE.CR` | Initial differential weight and crossover probability. SHADE adapts them during search. |
| `SHADE.c` | Number of adaptive `F`/`CR` memory channels. |
| `SHADE.p` | Fraction of the best population eligible for current-to-pbest mutation. |
| `PRE_OPT` | Enables a one-time initial SHADE-style pre-optimization only when a separate `DE` block is also present. The provided prime YAML files do not define `DE`, so `PRE_OPT` alone currently has no effect. |

The script also recognizes optional blocks that are absent from the supplied
prime YAML files:

| Optional block | Required fields | When used |
| --- | --- | --- |
| `DE` | `iters`, `F`, `CR`, `c`, `p` | One initial SHADE-style pass when `PRE_OPT` is true. Despite the key name, it calls `differential_evolution_system_shade()`, not the plain DE method. |
| `SA` | `T`, `T_min`, `step`, `alpha`, `iter`, `ptresh` | Simulated annealing after each local-optimization stage. |
| `GA` | `iters`, `elitism_rate`, `mutation_rate`, `mutation_strength` | Genetic search after each local-optimization stage. |

When present, `SA`, `GA`, and `SHADE` are run sequentially in that order at
the end of a stage; they are not mutually exclusive selector values.

### Consistency checklist

- `len(WAVELENGTHS) == len(WAVEWEIGHTS)` and `0 <= P_WAVE < len(WAVELENGTHS)`.
- `STRUCTURE`, `SURF_TYPE`, `MAT_TYPE`, and `MAT_CATA` have the same number of pipe-separated entries.
- `CFG_NUM == 1` for the current prime script.
- `len(EPOCH) == len(ADD_PARAM) == len(OPT_MAT_STAGE)`.
- Surface IDs in `SURF_GAP`/`ROC` refer to the expanded physical system, including the inserted stop—not directly to `STRUCTURE` entries.

## Outputs

Each seed receives a timestamped subdirectory. The best candidates are written under `min_loss_*`, including prescriptions and analysis figures.

The script attempts a large optional GPU allocation before initialization. Failure is caught, but users should still review memory requirements before launching a large candidate batch.
