# Micro Catadioptric Systems

Source directory: `test_cake/`

These examples design and evaluate compact catadioptric systems with refractive, reflective, obscured, and Q-type surfaces. They also demonstrate nested assembly tolerances and digital compensation.

## Structure generation

| Script | Purpose |
| --- | --- |
| `test_gen_cake.py` | Generates the nominal micro catadioptric structure. |
| `test_gen_cake_inv.py` | Generates the inverse structural variant. |

```bash
cd test_cake
python test_gen_cake.py --name gen_cake
python test_gen_cake_inv.py --name gen_cakex_inv
```

The generation code uses a customized Delano optimization, then constructs the physical `System` explicitly with `Qbfs` surfaces. Special aperture modes model central obscuration, and `MIRROR` materials create reflective interfaces. In the checked-in implementations, the nominal script orders the refractive pair before the two reflective surfaces, whereas the `_inv` script constructs the two reflective surfaces first and the refractive pair afterward. Thus “inverse” changes the hard-coded optical ordering; it is not a flag read from YAML.

## Nested tolerance and compensation

`test_cake_fix.py` loads a finished catadioptric prescription, creates multiple tolerance dictionaries, and calls `ini_tol_sys()` repeatedly to construct nested packages.

At each training stage it:

1. samples progressively larger decenter, tilt, and thickness errors;
2. applies quick focus as optical compensation;
3. renders PSFs, chief-ray centers, and relative illumination;
4. simulates degraded RGB images;
5. optionally concatenates two field-coordinate channels;
6. trains the configured restoration network;
7. reports PSNR, SSIM, and LPIPS.

```bash
cd test_cake
python test_cake_fix.py
```

This command is a workflow template, not runnable unchanged: the checked-in
`cake_fix.yaml` and `cakex_fix.yaml` set `FILE: './'`. Replace it with the JSON
prescription produced by the corresponding generation run (or another
compatible finished prescription), and provide the external dataset selected
by `DATASET`, before starting training.

## Offline simulation

`test_cake_sim.py` provides `simulate_images()` and `process_images()` helpers for evaluating blur, distortion, Zernike errors, and restoration. Placeholder paths must be replaced before use.

The selected `result_path` must already exist before the initial analysis is
saved. The script explicitly requires an even `VALID_PATCH_SIZE` and odd
`PSF_SIZE` and `PSF_SAMPLING`; it raises `ValueError` otherwise. Supply a
checkpoint compatible with `Model.load()` and the configured `NET`.

## Configuration families

The directory includes nominal and inverse generation files plus fixed-system training files such as `cake_fix.yaml` and `cakex_fix.yaml`. These configurations control system geometry, PSF sampling, tolerance scales, restoration model, dataset, and training schedule.

## Generation configuration reference

The `gen_*.yaml` files drive `test_gen_cake.py` or
`test_gen_cake_inv.py`. These scripts construct a fixed four-surface
catadioptric topology in code; there is no general topology string in the
YAML.

| Key | Meaning in the generation scripts |
| --- | --- |
| `DEVICE` | PyTorch device. The prescription optimizer uses fused Adam and is intended for CUDA. |
| `SEED` | List of complete independent generation runs. |
| `WAVELENGTHS`, `WAVEWEIGHTS`, `P_WAVE` | Wavelengths in millimeters, their merit weights, and zero-based primary-wavelength index. |
| `MAX_VIEW`, `NORM_VIEWS`, `AZIMUTHS` | Maximum half-field in degrees and the normalized fields/azimuths used for optimization and reporting. |
| `DELANO_SYS` | Number of first-order candidates generated before selecting one seed structure. |
| `LR_DELANO` | Adam learning rate in the customized first-order Delano refinement. |
| `MAT` | Material name for the refractive part. The script looks in the glass catalog first, then the plastic catalog. |
| `SYS_NUM` | Number of full Q-bfs prescriptions optimized in parallel after first-order selection. |
| `CFG_NUM` | Configuration-axis length. The provided generation scripts expect `1`. |
| `SAMP_METHOD` | Default pupil sampling distribution stored on the constructed `System`. |
| `SAMP_RAYS` | Merit-function ray sampling order. |
| `MERIT` | Standard `Merit.forward_loss()` operands. Surface IDs in `SURF_GAP` refer to the explicitly constructed physical surfaces. |
| `SHADE` | Stage-end SHADE search: iterations, initial `F`/`CR`, adaptive-memory count `c`, and p-best fraction `p`. |
| `EPOCH` | Adam iterations in each surface-optimization stage. |
| `ADD_PARAM` | Parameters unfrozen at each corresponding stage. `conic` and `qiN` names are used by the provided Q-bfs surfaces. |
| `LR_OPT` | Base learning rate for curvature, thickness, conic, and Q coefficients. |
| `DROPOUT` | Candidate-level gradient dropout probability. |
| `OPTIC_UPDATE_FREQ` | Interval for invalid-candidate replacement and the scripts' custom aperture/focus update. |
| `SAVE_FREQ` | Interval for intermediate analysis output. |

`cake`, `cakex`, and `cake135` are project codenames for different system
specifications. Their names, including the number `135`, do not have a formal
parameter meaning and are not parsed by the code. The `_inv` suffix is the
exception: it denotes the inverse hard-coded ordering described above.
`len(EPOCH)` must equal `len(ADD_PARAM)`.

## Tolerance-training configuration reference

The `*_fix.yaml` files drive `test_cake_fix.py`.

| Key | Meaning in the training script |
| --- | --- |
| `FILE` | Path to a finished compatible HappyLens JSON prescription. The checked-in `./` placeholder must be replaced. |
| `SYS_NUM` | Parallel nominal systems loaded from the prescription; the examples use one. |
| `CFG_NUM` | Number of parallel copies along the system's single configuration axis. In this fixed-system script the three copies receive independently sampled tolerance errors; they are not zoom or focus positions. |
| `SAMP_METHOD`, `SAMP_RAYS` | System pupil distribution and geometric merit sampling order.<br>`SAMP_RAYS` is passed to `Merit`; the initial report uses `save_analysis_results()`'s default sampling argument. |
| `TOL_DECENTER` | Final standard deviation/scale for sampled x/y package decenter, in millimeters. |
| `TOL_TILT` | Final sampled package-tilt scale, in degrees. |
| `TOL_THICK` | Final sampled surface-spacing scale, in millimeters. |
| `PSF_ANG_NUM`, `PSF_AZI_NUM` | Random field and azimuth counts per training iteration. Training batch size is their product times `SYS_NUM × CFG_NUM`. |
| `PSF_SAMPLING`, `PSF_SIZE`, `PSF_DELTA` | Pupil sampling order, square PSF size, and image-plane pitch in micrometers. |
| `NOISE_G`, `NOISE_P` | Gaussian and Poisson noise parameters passed to `simulate_rgb()`. |
| `RL_SAMPLING` | Pupil sampling order used for relative illumination. |
| `RENDER_R` | Millimeter normalization radius for the optional two field-coordinate channels used only by model names ending in `+F`. |
| `WAVEWEIGHTS_R/G/B` | Wavelength-to-RGB PSF mixtures. |
| `IMG_WEIGHT` | Multiplier on the network image loss. |
| `NET` | Case-sensitive `nets.Model` selector. `+F` variants receive five input channels; ordinary variants receive RGB only. |
| `NET_PTH` | Optional checkpoint loaded through `Model.load()`; `null` starts from random weights. |
| `DATASET` | Dataset name below `../../Data/`, with expected `train/` and `valid/` children. |
| `EPOCH`, `LR_NET` | Network training epochs and AdamW learning rate. |
| `TRAIN_PATCH_SIZE`, `VALID_PATCH_SIZE` | Training and validation image crop sizes in pixels. |

Tolerance magnitudes ramp linearly from `1/EPOCH` to their configured values
during training. Validation samples the full configured magnitudes. The script
uses two nested package levels: individual surfaces `1`–`4`, then a package
covering surfaces `1`–`2`; its range keys therefore assume the four-surface
prescription produced by these generation workflows.

Use an odd `PSF_SIZE` (and normally an odd `PSF_SAMPLING`) so the convolution
crop is symmetric. The supplied values satisfy this requirement.
