# Rays and Ray Tracing

## `Ray`

```python
Ray(o, d, wavelength)
```

`Ray` is a mutable geometric-ray container.

| Attribute | Shape | Meaning |
| --- | --- | --- |
| `wavelength` | `[W]` | Wavelengths in millimeters. |
| `o` | `[W,...,3]` | Current positions. |
| `d` | `[W,...,3]` | Current normalized directions. |
| `t` | `[W,...]` | Accumulated optical path length. |
| `chief_id` | usually `[W,sys,cfg,ang,azi]` | Chief-ray index for each bundle. |
| `valid` | `[W,...]` bool | Validity mask. |

The constructor accepts scalar or vector wavelengths and copies the input positions and directions across the wavelength dimension. It does not normalize `d`; call `normalize(d)` before construction when needed.

## Standard batch layout

For rays created by `System.sample_ray_2d()`:

```text
[wavelength, system, configuration, field, azimuth, pupil_ray, xyz]
```

This layout enables simultaneous differentiation over many wavelengths, candidate systems, configurations, and fields. A method documented as accepting a scalar field often converts it into a one-element field dimension rather than removing the dimension.

## Pupil sampling

`pupil_distribution(rays_h, rays_w, distribution)` returns normalized points in `[-1,1]²` and prepends a pupil-center sample used as the initial chief ray.

- `square` produces a regular rectangular grid and is required by PSF routines that reshape rays into a pupil image;
- `hexapolar` maps a rectangular grid to a hexapolar-style pupil distribution;
- `fibonacci` produces a Fibonacci disk distribution;
- `ring` produces concentric rings and is useful for layouts and spot diagrams;
- `line` samples a single normalized pupil diameter.

The `sampling` argument is not always the final ray count. With
`System.sample_ray_2d(sampling=M)`, `square`, `hexapolar`, and `fibonacci`
produce `M² + 1` entries including the prepended center; `line` produces
`M + 1`; and `ring` produces `2 + 3M(M+1)` because the implementation
also prepends a center to the center already created by the ring sampler.

`System.sample_ray_2d()` scales these coordinates by entrance-pupil diameter, applies optional vignetting, rotates them for each azimuth, and constructs finite- or infinite-conjugate ray bundles.

## Intersection solver

```python
Solver(MAXITER, TOLERANCE_TIGHT, TOLERANCE_LOOSE, METHOD, surf_samp)
```

`Solver.solve(surf, ray, mode='forward')` dispatches to one of three numerical methods:

| Method | Characteristics |
| --- | --- |
| `contraction()` | Fixed-point/contraction iteration intended only for small fields of view. |
| `newton()` | Uses surface first derivatives; default for `Sphere`. |
| `halley()` | Uses first and second derivatives for higher-order updates. |

The default attached to a surface is equivalent to `Solver(20, 1e-11, 1e-9, 'newton', 512)`. Failed or physically invalid intersections are tracked through the ray validity mask.

## Per-surface propagation sequence

`Sphere.propagate()` performs the following operations:

1. Save the incoming local position and direction.
2. Solve the ray/surface intersection.
3. Compute the local normal.
4. Apply reflection or vector Snell refraction.
5. Apply the clear-aperture test.
6. Accumulate physical optical path length in `ray.t`.
7. Translate the ray to the next local reference plane.

`System.propagate()` repeats this sequence and handles `PACKAGE` objects transparently.
It currently accumulates the intermediate position/direction tensors even
when called with `record=False`; that flag controls the returned object, not
the peak tracing memory required.

## Chief rays

Chief-ray indices are determined from the sampled bundle near the pupil center. With `System.vig_chief=True`, the propagated pupil coordinates are used to select the ray closest to the valid-bundle centroid. Analysis and PSF routines use `chief_id` to define image centers and reference optical path lengths.

## Differentiable PSF operators

### `CoherentPsfOp`

```python
CoherentPsfOp.apply(o, d, grid, opd, k)
```

Computes coherent plane-wave accumulation on an image grid. The custom backward implementation provides gradients with respect to ray positions, directions, image grid, and optical path difference; wavenumber is treated as constant.

### `RayleighSommerfeldPsfOp`

```python
RayleighSommerfeldPsfOp.apply(o, t, grid, k, l)
```

Computes Rayleigh–Sommerfeld propagation from exit-pupil ray samples to the image grid. Its backward implementation differentiates with respect to ray positions, optical path length, and grid coordinates.

Normal users should call these operators through `Merit.psf_co()` or `Merit.psf_rs()` rather than using `.apply()` directly.

## Forward versus reverse tracing

Forward tracing moves from object to image and accumulates optical path. Reverse tracing is used mainly to estimate pupil ranges and back-project from image space. Reverse propagation intentionally omits some forward-only bookkeeping and is not a drop-in replacement for computing a reverse optical path.
