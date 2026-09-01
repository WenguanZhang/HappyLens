# Lens Component Deletion

Source directory: `test_del/`

This application explores lens simplification by identifying a low-contribution component, continuously flattening/collapsing it, deleting its surfaces, and recovering image quality.

## Optical deletion

`test_component_del.py` is the optical-only entry point. The default prescription is selected by the `name` variable.

```bash
cd test_del
python test_component_del.py
```

Main loop:

1. Load a reference prescription and save its initial analysis.
2. Freeze prescription-specific parameters.
3. Call `Deletion.find_del_surfs()` to select an element interval.
4. Combine standard optical loss with `del_surf_loss()`.
5. Optimize until the sag/spacing deletion threshold is met.
6. Call `System.del_surfs()` and quick-focus the simplified system.
7. Save the final JSON and analysis figures, then export the JSON to Zemax.

### Deletion stopping condition

The collapse test intentionally uses logical **or**: a component is deleted
when either the maximum internal separation residual or the maximum sag
residual falls below its scaled threshold. The two residuals are alternative
indicators that the selected component has collapsed; they are not required
to pass simultaneously. After a successful deletion, the loop continues
until `DEL_NUM` components have been deleted.

### Optical-deletion YAML

The files without `_ref` or `_co` suffixes configure
`test_component_del.py`.

| Key | Meaning in this script |
| --- | --- |
| `DEL_NUM` | Number of components to delete. After each successful structural deletion, the script selects another candidate component until this count is reached. |
| `DEL_THRESH` | Dimensionless threshold factor. The script multiplies it by twice the larger clear-aperture radius at the two endpoints of the selected interval, producing a millimeter threshold. |
| `DEL_WEIGHT` | Weight of `reses_loss + flats_loss`, the continuous collapse/flattening objective. |
| `OPT_WEIGHT` | Weight of the ordinary deletion-aware optical merit. |
| `LR_OPT` | Adam learning rate used by `Deletion.params_lr()`. |
| `OPTIC_UPDATE_FREQ` | Interval for material fitting, aperture maintenance, paraxial refresh, and quick focus. |
| `SAVE_FREQ` | Interval for intermediate analysis directories. |
| `PERTURB_SCALE` | Present in the checked-in files but not read by `test_component_del.py`. |
| `RENDER_R` | Present in the checked-in files but not read by `test_component_del.py`. |

## Network-only compensation after component deletion

`test_del_ref_raw.py` starts from a lens whose selected component has already
been deleted. The remaining lens parameters stay fixed, and only the digital
restoration network is trained. The script computes PSFs and relative
illumination for the component-deleted lens, simulates RAW degradation, runs
the RAW ISP, and optimizes the selected network.

```bash
cd test_del
python test_del_ref_raw.py --name g_014
```

> **DeepSN authorization notice:** The checked-in `_ref.yaml` examples select
> `DEEPSN` by default. HappyLens has not received explicit permission to
> redistribute the DeepSN-Net implementation, so the implementation is not
> included here. Visit the authors' repository for the upstream project and
> its current usage terms:
> <https://github.com/pandazcx/DeepSN-Net>. Alternatively, change `NET` to a
> bundled model compatible with this workflow, such as `FSNET`, and review
> tensor/output handling when changing model families.

## Joint lens-network optimization after component deletion

`test_del_co_raw.py` also starts from a lens whose selected component has
already been deleted, but it jointly optimizes the remaining lens parameters
and the digital restoration network. This is the optical-digital co-optimization
counterpart to the network-only `test_del_ref_raw.py` baseline.

```bash
cd test_del
python test_del_co_raw.py --name g_014
```

Checked-in configuration families include `g_014`, `g_015`, `l_004`, and
`l_022`, with `_ref.yaml` and `_co.yaml` variants. With the currently active
Schott catalog, `g_014`, `l_004`, and `l_022` are directly loadable. The
`g_015` prescriptions reference `LAH52`, which is not in that active catalog;
they are reference inputs until material-catalog handling is adapted. This
limitation affects the optical-only, network-only, and joint entry points.

The checked-in `_co.yaml` examples also select the unauthorized-for-
redistribution `DEEPSN` placeholder. HappyLens does not include that
implementation; refer to <https://github.com/pandazcx/DeepSN-Net> or set `NET`
to a bundled implementation before running the examples. See
[Imaging and Networks API](../nets_api.md) for available model names and
[Third-Party Network Implementations](../third_party.md) for licensing status.

## RAW-compensation YAML

The `_ref.yaml` and `_co.yaml` families share these imaging parameters:

| Key | Meaning |
| --- | --- |
| `WAVEWEIGHTS_R/G/B` | Spectral mixtures used by `psf_to_rgb()` for the three output channels. |
| `PSF_ANG_NUM`, `PSF_AZI_NUM` | Random normalized-field and azimuth samples per training iteration. The batch size is their product times `SYS_NUM × CFG_NUM`. |
| `PSF_SAMPLING` | Square-pupil sampling order used by `psf_rs()`. |
| `PSF_SIZE` | PSF kernel height and width; loaders request patches enlarged by `PSF_SIZE - 1`. |
| `PSF_DELTA` | Image-plane PSF pitch in micrometers. |
| `NOISE_G`, `NOISE_P` | Gaussian and Poisson noise parameters passed to `simulate_raw()`. |
| `RL_SAMPLING` | Pupil sampling order for relative illumination. |
| `NET` | Case-sensitive restoration-model selector. The checked-in DeepSN choice is an unavailable placeholder; see the notice above. |
| `DATASET` | Dataset directory name below `../../Data/`; the script expects `train/` and `valid/` children. |
| `EPOCH`, `LR_NET` | Training epochs and AdamW network learning rate. |
| `TRAIN_PATCH_SIZE`, `VALID_PATCH_SIZE` | Final RAW/RGB crop sizes in pixels. |

`test_del_ref_raw.py` uses only those network/imaging fields and never creates
an optical optimizer: the `_ref.json` prescription is fixed. Its checked-in
`RENDER_R` value is not read because this RAW workflow has no field-coordinate
channel branch.

`test_del_co_raw.py` additionally reads `MERIT`, `OPT_WEIGHT`, `IMG_WEIGHT`,
`LR_OPT`, and `OPTIC_UPDATE_FREQ`. It differentiates the imaging loss through
the PSF into the remaining lens parameters as well as updating the network.
In the current script, `PERTURB_SCALE`, `RENDER_R`, `MAX_RADIUS`, and
`SAVE_FREQ` are present in `_co.yaml` but do not control execution; analysis is
saved once per epoch, and periodic optical maintenance explicitly disables
both aperture updates and quick focus.

## Simulation utility

`test_simulate_raw.py` contains higher-level `simulate_images()` and `process_images()` helpers for image simulation, optional distortion, Zernike error, and restoration. It currently contains placeholder YAML/data/checkpoint paths and requires user adaptation before execution. Create the selected `result_path` before the initial analysis is saved. The script explicitly requires an even `VALID_PATCH_SIZE` and odd `PSF_SIZE` and `PSF_SAMPLING`.

## External requirements

The RAW examples expect HDF5 datasets outside the repository, pretrained or newly trained restoration networks, TorchMetrics, and enough GPU memory for optical and image batches. Dataset paths currently follow `../../Data/<DATASET>/...` relative to `test_del`.
