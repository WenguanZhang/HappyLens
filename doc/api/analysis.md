# Analysis

```python
analysis = lens.Analysis(system)
```

`Analysis` owns a reference to one `System`. Most methods are decorated with `torch.no_grad()` and are intended for visualization, validation, and report generation rather than optimization.

## Layout and ray inspection

### `single_ray_trace(sys_id, cfg_id, ray)`

Traces a bundle expected to contain a single ray of interest and prints its field, wavelength, surface intersections, directions, and path information.

### `sample_ray_1d(sampling, wavelength)`

Creates meridional rays for 2D layout plots. This helper follows the system field and pupil conventions.

This helper and `plot_setup_with_trace()` access the flat entries of
`System.system` directly and are intended for systems without tolerance
`PACKAGE` wrappers. Remove the wrappers before using these layout routines.

### `plot_setup_with_trace(sys_id=0, cfg_id=0, M=3)`

Draws surface profiles and a small set of traced rays. `M` controls the one-dimensional pupil sampling density. The function creates Matplotlib figures and does not return a numerical metric.

## Point-spread functions

### `psf(...)`

```python
psf = analysis.psf(
    sys_id,
    cfg_id,
    pupil_samp,
    image_samp,
    image_delta,
    norm_view,
    azimuth,
    wavelength=None,
    split_channel=False,
    show=True,
)
```

Computes a diffraction PSF using propagated optical path lengths.

| Argument | Meaning |
| --- | --- |
| `pupil_samp` | Square pupil-grid width. An odd value is normally used. |
| `image_samp` | Square output-grid width. An odd value is normally used.|
| `image_delta` | Image-plane pixel pitch in um. |
| `norm_view` | Field normalized to `system.max_view`. |
| `azimuth` | Field azimuth in degrees. |
| `wavelength` | Optional single wavelength in millimeters. `None` combines system wavelengths. |
| `split_channel` | Keep individual wavelength PSFs instead of combining them. |
| `show` | Draw a heat map. |

The return value is `[H,W]` when wavelengths are combined and `[wav,H,W]`
when `split_channel=True`. Each returned wavelength plane is normalized by its
own sum; the combined result uses normalized `system.waveweights`.

### `psf_spot(...)`

Uses bilinear scattering of geometric ray intersections instead of diffraction propagation. Arguments follow `psf()`. This is useful for fast diagnostics but does not model diffraction.

Its return shapes follow `psf()`: `[H,W]` normally or `[wav,H,W]` with
`split_channel=True`.

## Wavefront

```python
opd = analysis.wavefront(
    sys_id,
    cfg_id,
    pupil_samp,
    norm_view,
    azimuth,
    wavelength=None,
    use_exit_pupil_shape=True,
    show=True,
)
```

Computes optical path difference relative to the chief ray and an exit-pupil reference sphere. With `use_exit_pupil_shape=True`, valid exit-pupil samples are triangulated and interpolated onto a normalized square grid. The plotted wavefront is expressed in waves. The method returns a one-dimensional NumPy array of valid OPD samples in millimeters, excluding the sampler's prepended center; it does not return the interpolated wavefront grid shown in the plot.

## MTF

```python
frequency, tangential, sagittal = analysis.mtf(
    sys_id,
    cfg_id,
    pupil_samp,
    image_samp,
    image_delta,
    norm_view=None,
    azimuth=None,
    wavelength=None,
    freq_max=None,
    freq_delta=None,
    show=True,
)
```

The PSF is Fourier transformed and sampled along orthogonal axes. `frequency` is in lp/mm. `norm_view=None` and `azimuth=None` use all values stored in the system. The output arrays are `[field,azimuth,frequency]` for the selected candidate/configuration.

The Nyquist frequency is `1 / (2 * image_delta_mm)`. A requested `freq_max` above this limit raises an exception.

## Geometric image quality

### `spot_diagram(sys_id=0, cfg_id=0, sampling=7, samp_method='ring')`

Draws a subplot for every stored normalized field and azimuth. Wavelengths are shown separately, and titles report RMS spot radius in micrometers. The Airy radius is overlaid for scale.

### `distortion(sys_id, cfg_id, pupil_samp, field_samp, wavelength=None, show=True)`

Uses an f-tan(theta) paraxial reference and compares traced chief-ray image height against the ideal height. Returns a wavelength-dependent distortion tensor.

The returned tensor keeps all traced wavelengths and the singleton azimuth
used internally, with shape `[wav,sys,cfg,field,1]`; `sys_id` and `cfg_id`
select what is plotted but do not slice the returned tensor.

### `relative_illumination(sys_id, cfg_id, pupil_samp, field_samp, wavelength=None, show=True)`

Estimates relative illumination from the area occupied by valid direction samples and normalizes by the on-axis value. The implementation is an approximation and should be interpreted accordingly.

The return value is a one-dimensional tensor of length `field_samp` for the
selected candidate and configuration.

When `wavelength=None`, this `Analysis` method traces every configured
wavelength but currently evaluates illumination from wavelength index `0`, not
from `system.p_wvl`. Pass an explicit wavelength when that distinction matters.
The optimization-side `Merit.relative_illumination()` instead defaults to the
primary wavelength.

## Saving a report

```python
analysis.save_analysis_results(
    path,
    sys_id=0,
    loss=0.0,
    samp_rays=3,
    samp_method="ring",
)
```

Saves the selected prescription as JSON and writes layout and spot-diagram SVG files for every configuration. The caller must create `path` before calling this method.

## Analysis versus optimization APIs

Use `Analysis` when you need a human-readable plot or a detached numerical report. Use `Merit` when gradients, all candidate systems, or all configurations must be preserved. The two classes deliberately expose related PSF, distortion, and illumination functions with different output shapes.
