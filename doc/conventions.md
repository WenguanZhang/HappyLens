# Conventions

## Units

| Quantity | API unit | Example |
| --- | --- | --- |
| Length, radius of curvature, and thickness | mm | `roc=8.0` |
| Wavelength | mm | `550e-6` = 550 nm |
| Maximum field and azimuth | degrees | `max_view=35.0` |
| Normalized field | dimensionless | `norm_view=1.0` |
| PSF image-plane sampling interval | μm | `image_delta=0.5` |
| MTF spatial frequency | lp/mm | `freq_max=500` |

## Coordinates and angles

- The optical axis is the z axis.
- `norm_view * max_view` gives the physical half-field angle.
- `azimuth=0` corresponds to the y-z meridional plane; azimuth follows the x/y rotation convention implemented in the source.
- The `roc` field in a JSON prescription is a radius of curvature. The `Sphere.roc` object attribute stores its reciprocal, the surface curvature.

## Ray tensors

```text
[wav, sys, cfg, ang, azi, ray, xyz]
```

- `wav`: wavelength;
- `sys`: parallel candidate system;
- `cfg`: the framework's single configuration axis. In a prime/fixed-focal-length system it may index discrete operating conditions; in a zoom or focusing system it indexes the zoom/focus structural states;
- `ang`: field sample;
- `azi`: azimuth sample;
- `ray`: pupil ray;
- `xyz`: three position or direction components.

## PSF tensors

```text
[wav, sys, cfg, ang, azi, image_y, image_x]
```

After `Merit.psf_to_rgb()` the first dimension becomes RGB:

```text
[rgb, sys, cfg, ang, azi, image_y, image_x]
```

## Numerical precision and devices

Optical experiments normally use `torch.float64`, while neural networks normally use `torch.float32`. Many tensors created by the current implementation inherit the PyTorch default device, so configure the device before constructing a `System`.

```python
torch.set_default_dtype(torch.float64)
torch.set_default_device("cuda:0")
```

These settings are process-global. Explicitly convert dtypes when optical simulation and neural image processing share one process.

## Configuration-axis limitation

There is no separate tensor axis for “structural state” and “operating
condition.” A fixed-focal-length system may therefore evaluate several
discrete conditions through `cfg_num`. A zoom or focusing prescription already
uses `cfg` for its zoom/focus states; to evaluate another independent condition
for that structure, run it separately rather than trying to form a Cartesian
product inside one `System`.
