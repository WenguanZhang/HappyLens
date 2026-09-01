# Configuration and File Formats

## YAML configuration

`lens.GetYaml(path)` maps every top-level YAML key to an attribute of the returned object. Configuration files generally contain the following groups.

There is no repository-wide YAML schema and `GetYaml` does not reject unknown
or unused keys. Each entry-point script decides which attributes it reads.
Copy a configuration from the same workflow you intend to run; the presence
of a key in another YAML file does not imply that the current script uses it.

The structure-generation examples use additional topology and staged-search
parameters. See the detailed references for
[prime-lens generation](gallery/prime_generation.md#configuration-parameter-reference)
and [zoom-lens generation](gallery/zoom_generation.md#configuration-parameter-reference).

### Runtime

| Key | Description |
| --- | --- |
| `DEVICE` | Default PyTorch device, for example `cuda:0`. |
| `SEED` | Random seed. |
| `MATERIAL_CATALOG` | Material catalogs loaded before constructing a lens.<br>`GLASS` and `PLASTIC` each accept one catalog name, an ordered list of names, or an explicit JSON path. The two keys describe how the loaded data is used; they do not restrict catalog vendors. |

Each entry-point calls the catalog loader immediately after reading its YAML:

```python
cfg = lens.GetYaml(config_path)
lens.configure_material_catalog(cfg.MATERIAL_CATALOG)
```

If multiple glass catalogs are listed, they are merged from left to right. A
material name already supplied by an earlier catalog is not overwritten by a
later catalog; for example, `[schott, ohara]` gives Schott priority for any
duplicate name. A name such as `nikon` is resolved dynamically to
`lens/glass_nikon.json` (an exact `lens/nikon.json` file is accepted too), so
users can add catalogs without editing Python choices. An explicit relative
path (resolved from the current working directory) or absolute JSON path is
also accepted. `PLASTIC: hoya`, for example, loads the
Hoya JSON as the catalog used for surfaces marked `P`; the key does not require
the file to be named `plastic`. If the mapping or one of its entries is omitted,
the loader defaults to `schott` glass and the bundled `plastic` catalog.
Catalog names are case-insensitive. Loading another configuration updates the
process-wide catalog selection, so configure it before constructing `Delano`,
`Zoom`, or `System` objects and do not switch catalogs while existing systems
are in use. If a requested catalog cannot be found, loading stops with a
`FileNotFoundError` that identifies the `GLASS` or `PLASTIC` entry, the searched
paths, and the catalogs currently available; it never silently substitutes a
different catalog.

### Spectrum and field

| Key | Description |
| --- | --- |
| `WAVELENGTHS` | Wavelengths in millimeters. |
| `WAVEWEIGHTS` | Wavelength weights used by optical merit functions. |
| `P_WAVE` | Index of the primary wavelength. |
| `WAVEWEIGHTS_R/G/B` | Weights used to convert multi-wavelength PSFs to RGB. |
| `MAX_VIEW` | Maximum half-field angle; a multi-configuration system may use one value per configuration. |
| `NORM_VIEWS` | Normalized field samples. |
| `AZIMUTHS` | Azimuth samples in degrees. |

### System and tracing

| Key | Description |
| --- | --- |
| `SYS_NUM` | Number of candidate systems evaluated in parallel. |
| `CFG_NUM` | Length of the framework's single configuration axis.<br>For a prime/fixed-focal-length system it may represent parallel discrete operating conditions.<br>Zoom/focus structures already use this axis for structural states, so additional operating conditions must be simulated one at a time. |
| `PRE_SAMP` | Optional pre-sampling order for systems with strong pupil aberration. |
| `SAMP_METHOD` | Pupil distribution: `square`, `hexapolar`, `fibonacci`, `ring`, or `line`. PSF-grid routines explicitly request `square`; layout and spot workflows commonly use `ring`. |
| `SAMP_RAYS` | Sampling order used by optimization merit functions. |
| `FIX_RADIUS_SURF` | Surface indices whose clear-aperture radii remain fixed. |
| `MAX_RADIUS` | Optional upper bound applied when apertures are updated. |

### Merit function

`MERIT` is a mapping from operand name to operand parameters:

```yaml
MERIT:
  EFL: {target: 4.45, weight: 1.0}
  FNO: {target: 2.16, weight: 1.0}
  SPOT: {ref: rms, weight: 200.0}
  BFL: {target: 0.5, weight: 1.0}
  AIR_THICK: {target: 0.05, weight: 50.0}
```

Names are case-sensitive. The complete operand table is documented in [Optimization](api/optimization.md).

### PSF and computational imaging

| Key | Description |
| --- | --- |
| `PSF_ANG_NUM`, `PSF_AZI_NUM` | Number of field and azimuth samples per training batch. |
| `PSF_SAMPLING` | Pupil sampling order used to render PSFs. |
| `PSF_SIZE` | PSF image width and height. |
| `PSF_DELTA` | Image-plane sampling interval in micrometers. |
| `NET` | Model name accepted by `nets.Model`. |
| `LR_OPT`, `LR_NET` | Optical and neural-network learning rates. |
| `OPT_WEIGHT`, `IMG_WEIGHT` | Optical and image-loss weights. |

### Root `lens_yaml/` files

The root configurations are prescription and analysis configurations for the
interactive `test_basic` examples. They intentionally contain only fields
read by `test_basic.py` or `test_basic_zoom.py`: runtime/device and material-catalog settings,
spectrum and RGB wavelength weights, field/vignetting data, system dimensions,
merit operands, and the optical optimizer controls used by those scripts.
Network-training and dataset parameters belong in the workflow-specific YAML
files under `test_del/` and `test_cake/`; do not add them to a root YAML unless
a root entry point is also changed to consume them.

Current root prescription compatibility is narrower than the list of files in
those directories:

| Entry | Current use |
| --- | --- |
| `f_006`, `g_014`, `g_015`, `gauss`, `l_004`, `l_022`, `phone`, `wide_50` | Prime-lens pairs directly loadable by `test_basic/test_basic.py`; their YAML selects the required bundled catalogs. `g_015` merges Schott and Ohara in that priority order, while `phone` obtains `APL5014CL` and `POLYSTYR` from the plastic catalog. |
| `zoom_3x` | Multi-configuration pair used by `test_basic/test_basic_zoom.py`. |

An unknown catalog name raises a configuration error when catalogs are loaded.
An unknown material inside a valid catalog is detected later during system
construction. Check every prescription material before treating a custom pair
as an executable example.

## Lens JSON prescriptions

The top-level order is `OBJECT`, one or more `SurfaceN` entries, and `IMAGE`.

```json
{
  "OBJECT": {
    "material": "VACUUM",
    "distance": null
  },
  "Surface1": {
    "type": "Standard",
    "stop": false,
    "radius": 4.35,
    "material": "N-SK2",
    "roc": 8.12,
    "thick": 1.31,
    "conic": 0.0
  },
  "IMAGE": {
    "radius": 3.69
  }
}
```

The loader relies on insertion order and does not run a schema-validation
pass. Keep `OBJECT` first and `IMAGE` last, use only the supported surface
type strings, and mark exactly one optical surface as the stop. An unknown
`type` is currently skipped rather than rejected at the dispatch point, while
a missing stop leaves `stop_id` undefined. Validate edited JSON before a long
run.

### `OBJECT`

- `material`: normally `VACUUM`.
- `distance`: object distance in millimeters; `null` denotes an object at infinity. A list can define one value per configuration.

Do not mix `null` and finite distances in the same list. The current sampler
tests whether *any* entry is `null` and then follows the infinite-object branch
for the whole multi-configuration system.

### Optical surfaces

| Field | Description |
| --- | --- |
| `type` | `Standard`, `Asphere`, `Qcon`, `Qbfs`, or `Binary2`. |
| `stop` | Whether this is the aperture-stop surface. Exactly one stop is expected. |
| `radius` | Clear-aperture radius in millimeters. |
| `material` | Medium after the surface, or the special value `VACUUM` or `MIRROR`. |
| `roc` | Radius of curvature in millimeters; `null` denotes a plane. |
| `thick` | Axial distance to the next reference plane in millimeters. A list defines multiple configurations. |
| `conic` | Conic constant. |
| `aperture` | Optional aperture mode: `float` for `r <= radius`, `circ` for annular transmission `min_r < r < max_r`, or `obsc` for a blocked middle annulus with transmission at `r < min_r` and `max_r < r <= radius`. |
| `min_r`, `max_r` | Optional radial boundaries for `circ` and `obsc`; these are per candidate system, not per configuration. |
| `zoom_type` | Optional two-character zoom motion/parameter-sharing label (`FF`, `FM`, `MF`, or `MM`).<br>It is consumed by `MeritZ` and the zoom scripts; preserve generated values when editing zoom JSON. |

Additional fields by surface type:

- `Asphere`: `ai_list`;
- `Qcon` and `Qbfs`: `qi_list`, `rnorm`;
- `Binary2`: `ai_list`, `diff_order`, `pi_list`, `rnorm`.

### `zoom_type` caution

`zoom_type` is operational metadata, not merely a display label. The zoom
workflow uses its second character to decide whether a thickness gradient is
averaged across configurations; `MeritZ.fix_lens_loss()` selects exact `FF`
entries, `smooth_zoom_loss()` selects exact `MM` entries, and the zoom air-gap
loss classifies neighboring entries from these characters. Values produced by
`Zoom.lens_instance()` are therefore safest to keep unchanged. A hand-written
multi-configuration JSON with missing labels is still loadable for tracing,
but zoom-specific sharing and merit logic must not be assumed to work.

### `IMAGE`

`radius` is the image-plane radius in millimeters. A list may be used for a multi-configuration system.

For every configuration-valued list (`OBJECT.distance`, surface `radius` and
`thick`, `IMAGE.radius`), supply exactly `CFG_NUM` entries. Scalar values are
repeated automatically; list lengths are not checked explicitly before tensor
broadcasting and paraxial calculation.

## Vignetting dictionary

`System.sample_ray_2d(..., vig=...)` accepts:

```python
vig = {
    "VUY": [...],
    "VLY": [...],
    "VUX": [...],
    "VLX": [...],
}
```

The arrays clip the positive/negative coordinate boundaries of the normalized
pupil. Before azimuth rotation, the remapped ranges are
`x ∈ [-1 + VLX, 1 - VUX]` and `y ∈ [-1 + VLY, 1 - VUY]`.
Thus `VUX`/`VLX` mean upper/lower X, not left/right in the order shown above.
Each value is a normalized fraction; zero leaves that boundary unchanged.
One-dimensional arrays are shared by every configuration and align with the
requested field samples; two-dimensional arrays may provide `[cfg,field]`
values explicitly.
