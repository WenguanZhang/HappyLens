# Utilities

## Configuration

### `GetYaml(file_path)`

Reads the file and assigns every top-level YAML key as an object attribute. `read_yaml()` returns the raw text. A missing file currently prints a message instead of raising a dedicated exception.

### `configure_material_catalog(config=None, *, glass='schott', plastic='plastic')`

Loads the material catalogs used process-wide by surface construction and
material optimization. Pass the YAML `MATERIAL_CATALOG` mapping, or use the
keyword arguments directly. `GLASS` and `PLASTIC` may each be one name, an
ordered list, or an explicit JSON path. Relative paths are resolved from the
current working directory. A name is discovered from
`lens/glass_<name>.json` or `lens/<name>.json`; it is not checked against a
fixed vendor list. Either role may load any discovered catalog. Catalogs are
merged from left to right, and the first catalog wins when the same material
name has different parameters. The function returns the normalized active
selection as a mapping of lists.

Call it after `GetYaml()` and before constructing `Delano`, `Zoom`, or `System`.
Existing systems retain tensors and material data created from the previous
selection, so switching catalogs while they remain in use is unsupported.

## Reproducibility and gradients

| Function | Description |
| --- | --- |
| `set_random_seed(seed=0)` | Seeds Python, NumPy, and PyTorch generators. |
| `clip_gradient(optimizer,grad_clip)` | Clamps existing parameter gradients elementwise. |
| `rand_dropout(optimizer,prob=0.1)` | Randomly zeros whole candidate slices of existing gradients with probability `prob`. |
| `limit_var(var,var_min,var_max)` | Returns the elementwise distance outside `[var_min,var_max]`, and zero inside; it is a soft-constraint residual, not an in-place clamp or reparameterization. |

## Vector and list helpers

| Function | Description |
| --- | --- |
| `normalize(d)` | Normalizes the final vector dimension. |
| `length(d)` | Computes Euclidean length along the final dimension. |
| `quaternion_raw_multiply(a,b)` | Raw quaternion multiplication. |
| `factorial(n)` | Factorial helper used by Q-type polynomials. |
| `find_key(d,target,path='')` | Finds a nested dictionary key path. |
| `list_convert(lst)` | Collapses a one-element list, or a list whose entries are all identical, to that common value; otherwise returns the list unchanged. |
| `generate_normalized_numbers(n,vmin=-1,vmax=1)` | Generates `n` bounded values whose sum is exactly 1.<br>See the edge conditions below. |

For `generate_normalized_numbers()`, `n=1` returns `[1.0]` without checking
the supplied bounds. For `n>1`, ensure `n*vmin <= 1 <= n*vmax`; an infeasible
interval causes unbounded recursive retries rather than a validation error.

## Material helpers

### `nv_to_g1_g2(n, v, mat_cata='G')`

Maps refractive index and Abbe number to the continuous two-coordinate material representation used during optimization.

### `g1_g2_to_n(g1, g2, wavelength, mat_cata='G')`

Evaluates refractive index from continuous material coordinates.

Unlike public ray and system APIs, this low-level helper expects `wavelength`
in micrometers. `Sphere.refractive_index()` accepts millimeters and multiplies
by `1e3` before calling it. Calling `g1_g2_to_n()` directly with millimeters
therefore produces the wrong dispersion.

### `fit_get_mat_id(params, method='M', mat_cata='G')`

Finds a catalog material close to optimized continuous coordinates. `System.material_fit()` provides the surface-level interface and updates stored material data.

Catalog dictionaries and parameter tensors are exposed as module globals.
Treat them as read-only reference data and use `configure_material_catalog()`
rather than mutating them directly.

## Zernike functions

| Function | Description |
| --- | --- |
| `zernike_radial(n,m,rho)` | Radial polynomial. |
| `zernike_noll_to_nm(j)` | Converts Noll index to `(n,m)`. |
| `zernike_noll(j,rho,theta)` | Evaluates one Noll-ordered mode. |
| `zernike_wavefront(rho,theta,coeffs)` | Sums `{noll_index: coefficient}` modes. |

`Merit.psf_rs_err()` interprets coefficients as waves and converts them to optical path at the selected wavelength.

## Visualization and export

### `plot_loss_pie(losses, labels, valids, path)`

Writes a loss-composition plot for valid candidates. Lists must have matching
lengths, the validity mask must select at least one candidate, and the mean
losses must have a nonzero total because percentages divide by that total.

### `read_prime_json_to_zmx(json_file, zmx_file, wave, wt, p_wvl, norm_views, max_field=None, field='ang')`

Converts a prime lens HappyLens JSON prescription to a Zemax text lens file.
The caller supplies wavelengths, weights, primary-wavelength index, normalized
fields, and maximum field. Although the signature defaults `max_field` to
`None`, the current implementation multiplies it by every normalized field, so
a numeric value is required in practice.

### `read_zoom_json_to_zmx(json_file, zmx_file, wave, p_wvl, norm_views, max_angle)`

Exports a multi-configuration zoom prescription. Unlike the prime-lens exporter, the current signature does not accept wavelength weights or a field-mode selector.

Both exporters explicitly handle `Standard`, `Asphere`, `Qcon`, and `Qbfs`
surface dictionaries. They do not currently export `Binary2` diffraction data,
custom aperture masks, or tolerance/package ownership. Export is intended for
interchange and validation; verify surface types, material names, sign
conventions, multi-configuration data, coordinate breaks, and diffractive
terms in the target optical-design application.
