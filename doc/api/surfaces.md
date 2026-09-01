# Surfaces

Every physical surface is an `nn.Module`. Optical parameters use a leading candidate-system dimension and, where needed, a configuration dimension.

## `Sphere`

```python
Sphere(
    radius,
    material,
    roc,
    thick,
    conic,
    mat_cata=None,
    aperture="float",
    min_r=None,
    max_r=None,
)
```

`Sphere` implements a rotationally symmetric conic surface. Despite the class name, nonzero `conic` values are supported.

| Argument | Stored form |
| --- | --- |
| `radius` | `[sys,cfg]` clear-aperture radius. |
| `material` | One medium name per candidate system. The material is the medium after the interface. |
| `roc` | Input radius of curvature; stored internally as reciprocal curvature `[sys]`. `None` becomes zero curvature. |
| `thick` | `[sys,cfg]` axial distance to the next local reference plane. |
| `conic` | `[sys]` conic constant. |
| `mat_cata` | `G` for glass, `P` for plastic, or `None` for special media. |
| `aperture` | `float`, `circ`, or `obsc`; see the exact masks below. |
| `min_r`, `max_r` | Per-candidate radial boundaries used by `circ` and `obsc`. |

`mat_cata` is shared by the whole batched surface, not stored separately for
each candidate. Therefore every ordinary material in one `Sphere` must come
from the same catalog. Also do not mix an ordinary glass/plastic name with
`VACUUM` or `MIRROR` across candidates on the same surface: the current
special-medium checks are batch-wide (`"VACUUM" in material` / `"MIRROR" in
material`) and would make the ordinary candidates follow the wrong optical
branch. Use a homogeneous ordinary-catalog batch, an all-vacuum batch, or an
all-mirror batch for each surface.

The aperture modes implement these masks (the prepended chief ray is retained
explicitly by the current code):

- `float`: a conventional circular clear aperture, `r <= radius`;
- `circ`: an annular transmitting aperture, `min_r < r < max_r`;
- `obsc`: a middle-annulus obstruction, transmitting `r < min_r` or
  `max_r < r <= radius`.

Despite its name, `circ` does not mean the ordinary disk aperture; use
`float` for that case.

### Geometry

| Method | Description |
| --- | --- |
| `surface(x, y)` | Returns sag at the supplied coordinates. |
| `surface_d(x, y, vec=True)` | Returns first derivatives; vector mode returns `(dx,dy,dz)`. |
| `surface_dd(x, y, vec=True)` | Returns second derivatives. |
| `surface_sag(surf_samp)` | Samples sag radially from axis to clear aperture. |
| `inter_normal(x, y, mode='forward')` | Returns normalized interface normals. |

Coordinates normally follow the full ray broadcasting layout. The implementations insert wavelength, configuration, field, azimuth, and ray singleton dimensions around per-system parameters.

### Optical interaction

| Method | Description |
| --- | --- |
| `refractive_index(wavelength)` | Returns `[wav,sys]` refractive indices using Schott or Sellmeier coefficients. |
| `intersect(ray, mode='forward')` | Uses the attached `Solver` to move rays to the surface. |
| `refract(ray, pre_surf, mode='forward')` | Applies vector Snell refraction; `MIRROR` applies reflection. |
| `judge_valid(o)` | Computes the aperture-validity mask. |
| `propagate(ray, pre_surf, radius_flag=True)` | Intersects, refracts, accumulates optical path length, and moves to the next local reference plane. |
| `reverse_propagate(...)` | Performs the corresponding reverse operation without recording optical path length. |
| `abcd(pre_surf, wavelength)` | Returns `[sys,cfg,2,2]` paraxial refraction-plus-translation matrices. |

Total internal reflection marks rays invalid. Square-root arguments are clipped by a small `eps` to keep invalid candidate systems numerically representable.

### Thickness tolerances

- `thickness()` returns nominal thickness plus `thick_tol`.
- `update_thickness_tol(scale)` samples a Gaussian perturbation for each system/configuration.

## Even-aspheric surface: `Asphere`

```python
Asphere(ai_list, **sphere_arguments)
```

Adds even radial polynomial terms to the conic base. Coefficients are registered as `ai4`, `ai6`, and so on. `surface()`, `surface_d()`, and `surface_dd()` are overridden, while intersection and refraction are inherited.

`ai_list` is organized by coefficient first and candidate system second. For example, two coefficients over three systems are shaped like `[[a4_0,a4_1,a4_2], [a6_0,a6_1,a6_2]]`.

## Q-type surfaces

### `Qcon(qi_list, rnorm, **sphere_arguments)`

Adds Q-con polynomial terms on the normalized radial coordinate `u²=(x²+y²)/rnorm²`. `surface_d(..., vec=False)` is not supported.

### `Qbfs(qi_list, rnorm, **sphere_arguments)`

Uses a best-fit-sphere-oriented Q basis. `calculate_coeff()` prepares recurrence coefficients, and `calculate_S_dS(u2)` evaluates the basis sum and derivative. The class overrides sag, derivatives, and paraxial behavior where required.

`rnorm` is stored as one normalization radius per candidate system. Keep it consistent with the usable clear aperture.

As with `Qcon`, `surface_d(..., vec=False)` is not supported.

## Diffractive surface: `Binary2`

```python
Binary2(diff_order, pi_list, rnorm, ai_list, **sphere_arguments)
```

`Binary2` combines an `Asphere` substrate with a wavelength-dependent polynomial phase profile.

| Method | Description |
| --- | --- |
| `phase(x, y)` | Evaluates the phase profile. |
| `_phi(r2)` | Internal radial phase polynomial. |
| `_dphid(r2)` | Internal radial derivative. |
| `refract(...)` | Adds diffraction-order-dependent direction change. |
| `abcd(...)` | Returns a paraxial matrix including diffractive power. |

`diff_order` selects the diffraction order, `pi_list` stores phase coefficients, and `rnorm` normalizes radius.

The current `Binary2.refract()` and `Binary2.abcd()` implementations do not
support `material="MIRROR"`; both raise `NotImplementedError` for a
diffractive mirror.

## Boundary and coordinate modules

### `OBJECT(material, distance)`

Represents object space. `distance` is one value per configuration; `None` denotes infinity. `propagate(ray)` moves the input rays to the first local reference plane. The current refractive-index implementation supports `VACUUM` object space only and raises `NotImplementedError` for another object-space material.

### `IMAGE(radius)`

Stores image-plane radius and provides a zero-sag planar `surface(x,y)`.

### `Dummy(pre_surf, thick)`

Represents translation without refracting power. It implements thickness, material access, ABCD propagation, and a surface-compatible `propagate()`.

### `Coordinate(decenter, tilt, flag, **dummy_arguments)`

Applies a coordinate break. `decenter` contains x/y shifts; `tilt` contains rotational components. `flag` distinguishes entering from leaving the transformed coordinate frame.

### `PACKAGE(decenter, tilt, pack)`

Groups several surfaces under a shared rigid-body transform and may contain nested packages. `transform()` applies the coordinate change, `propagate()` processes the package as a system element, and `propagate_elem()` recursively handles its contents.

Package nesting is the basis of HappyLens assembly-tolerance modeling.
