# Start Here

HappyLens focuses on lens optimization and automated structure search, with computational-imaging tools for workflows that continue beyond the optical prescription. Its implementation is organized into the `lens` and `nets` Python packages.

## Feature map

| Goal | Recommended entry point |
| --- | --- |
| Load an existing lens prescription | `lens.System(..., file="...")` |
| Sample and trace rays | `System.sample_ray_2d()`, `System.propagate()` |
| Draw a lens layout or spot diagram | `lens.Analysis` |
| Compute wavefronts, PSFs, and MTFs | `Analysis.wavefront()`, `Analysis.psf()`, `Analysis.mtf()` |
| Search a prime lens from first-order variables | `lens.Delano`, `lens.Generation_Prime` |
| Search a zoom structure across configurations | `lens.Zoom`, `lens.Generation_Zoom` |
| Optimize a physical prescription | `lens.Merit`, `lens.MeritZ`, `lens.DLS` |
| Run population-based or stochastic search | `Merit.genetic_system()`, `Merit.differential_evolution_system()`, `Merit.simulated_annealing_system()` |
| Delete lens components automatically | `lens.Deletion` |
| Model nested decenter, tilt, and thickness errors | `System.ini_tol_sys()`, `System.rand_decenter_tilt_thick_param()` |
| Simulate RGB or RAW image degradation | `nets.simulate_rgb()`, `nets.simulate_raw()` |
| Run an image-restoration network | `nets.Model` |

## Documentation scope

This documentation covers the framework itself:

- `lens/`: rays, surfaces, systems, analysis, and optimization;
- `nets/`: image degradation, ISP, data loading, and neural networks;
- `lens_json/`: lens prescription files;
- `lens_yaml/`: runtime configuration files.

The `test_*` directories contain research workflow scripts. Several require
external data, generated inputs, CUDA, or documented path/configuration setup; they
should not all be assumed runnable immediately after cloning. The [Example
Gallery](gallery/introduction.md) records those prerequisites, while the API
reference is limited to reusable interfaces under `lens/` and `nets/`.

## Suggested reading order

1. Read [Conventions](conventions.md) for units and tensor layouts.
2. Follow [Quickstart](quickstart.md) to load and analyze your first lens.
3. Use the [Optical API](lens_api.md) as the reference for optical operations.
4. Read the [Imaging and Networks API](nets_api.md) when building an end-to-end imaging pipeline.

## Build the documentation

```bash
python -m pip install -r doc/requirements.txt
python -m sphinx -b html doc doc/_build/html
```

Alternatively:

```bash
make -C doc html
```

The generated entry page is `doc/_build/html/index.html`. The repository-level `.readthedocs.yaml` points to the same configuration and can be used directly by Read the Docs.
