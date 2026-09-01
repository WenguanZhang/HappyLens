# Imaging and Networks API

The `nets` package provides RGB/RAW image degradation, a compact ISP pipeline, dataset loaders, and restoration models. It can consume PSFs produced by `lens.Merit` in end-to-end optical and digital optimization workflows.

## Unified model wrapper

```python
import nets

model = nets.Model("RESTORMER")
prediction = model(input_tensor)
loss = model.loss(prediction, target)
```

### Available model names

| Name | Input | Output |
| --- | --- | --- |
| `SRCNN` | RGB image | RGB image |
| `RESTORMER` | RGB image | RGB image |
| `FSNET` | RGB image | Three-scale RGB list |
| `FSNET+F` | Five-channel image/field tensor | Three-scale RGB list |
| `DIFF` | Training: ground truth and degraded image | Initial prediction, predicted noise, reference noise |

### Unavailable model placeholders

| Reserved name | Upstream | Behavior |
| --- | --- | --- |
| `MIMOUNET`, `MIMOUNET+F` | [MIMO-UNet](https://github.com/chosj95/mimo-unet) | Raises `RuntimeError` |
| `DWDN` | [DWDN](https://github.com/dongjxjx/dwdn) | Raises `RuntimeError` |
| `CDWDN` | Derived from DWDN | Raises `RuntimeError` |
| `DEEPSN`, `DEEPSN+F` | [DeepSN-Net](https://github.com/pandazcx/DeepSN-Net) | Raises `RuntimeError` |

These names remain reserved for configuration compatibility, but their
implementations are not distributed because the respective upstream
repositories do not provide explicit redistribution licenses. Each placeholder
directory contains only `model.py` and `loss.py`; selecting a reserved name
through `nets.Model`, or instantiating a placeholder class directly, raises an
error containing the upstream URL. See
[Third-Party Network Implementations](third_party.md) for the complete status.

`Model.forward(*args)` and `Model.loss(*args)` forward arguments unchanged to
the selected model and loss object.

Model names are case-sensitive and should be chosen from the tables above.
An unknown or misspelled name raises `ValueError` during construction and lists
the accepted names. Reserved unavailable names remain valid selectors, but
raise the licensing `RuntimeError` described above.

`Model.save(directory, filename)` writes
`<directory>/model_<filename>.pt` as a checkpoint containing the wrapper's
complete state dictionary under the `model` key. `Model.load(path, device)`
loads that format. The training workflows use the same wrapper-level checkpoint
layout, so their `net.pt` files are also compatible with `Model.load()` when
the selected model name and architecture match.

## Image degradation

### `img_conv_mul(img, psf, mode='fft')`

- `img`: `[B, C, H, W]`
- `psf`: `[B, C, 2M+1, 2N+1]`
- `mode`: `fft` or `conv2d`
- returns `(blurred, cropped_reference)` after removing a border equal to the PSF radius.

Use odd PSF height and width of at least 3. The cropping code slices by the
integer half-size; a `1×1` kernel produces empty spatial slices because
Python treats `-0` as `0`.

### `simulate_rgb(gts, psfs, rl, sigma, lamb, mode='fft')`

Applies a per-sample RGB PSF, relative illumination, Gaussian noise, and Poisson noise.

| Argument | Shape |
| --- | --- |
| `gts` | `[B,3,H+2M,W+2N]` |
| `psfs` | `[B,3,2M+1,2N+1]` |
| `rl` | `[B]` or `[B,H+2M,W+2N]` |
| `sigma`, `lamb` | `[B,1,1,1]` |

Here `H×W` is the returned cropped size. If `rl` is a spatial map rather than
a per-image scalar, the current implementation crops it by the PSF radius too,
so its input shape must actually match `gts`: `[B,H+2M,W+2N]`.
`lamb` is used as a divisor in the Poisson sampling expression and must be
strictly positive; set `sigma` to zero if Gaussian noise is not wanted.

### `simulate_raw(raw, color, psf, rl, sigma, lamb, mode='fft')`

The function demosaics Bayer RAW data, applies channel-dependent blur, samples the result back through the CFA, and adds noise. `color` uses the integer order encoded by `RGBG`.

Returns `(raw_blur, raw_label, cropped_color_map)`.

`raw` and `color` are `[B,H+2M,W+2N]`; `psf` is
`[B,3,2M+1,2N+1]`; `sigma` and `lamb` are `[B,1,1]`. As with
`simulate_rgb()`, a spatial `rl` map must match the uncropped RAW dimensions,
while a scalar illumination input is `[B]`. `lamb` must be strictly positive.

## ISP pipeline

```python
rgb = nets.isp(raw, rlc, color, wb, cm, alpha=None)
```

Pipeline: lens-shading correction → guided-filter denoising → white balance → demosaic → color correction matrix → Rec.709 gamma → tone mapping.

| Function | Operation |
| --- | --- |
| `isp_raw(raw, rlc, color, wb)` | Runs through demosaicing only. |
| `isp_rgb(rgb, cm, alpha=None)` | Applies CCM, gamma, and tone mapping to RGB. |
| `lsc(raw, rlc)` | Multiplies RAW data by a shading-correction gain. |
| `denoise(raw)` | Guided filtering on the four Bayer sub-lattices. |
| `white_balance(raw, color, wb)` | Applies four CFA gains. |
| `demosaic(raw, color, method='Malvar')` | Converts `[B,H,W]` RAW to `[B,3,H,W]`; supports `Malvar` and `Bilinear`. |
| `ccm(rgb, cm)` | Applies a batch of 3×3 color matrices. |
| `gamma(image, gamma_type='Rec709')` | Supports `Rec709` and `2.2`. |
| `tone_mapping(image, alpha=None)` | Applies the implemented quadratic tone curve twice. |

## Data loading

### `GetRGB`

Reads HDF5 files containing an `img` dataset in `[H,W,C]` order and returns random square crops. `train_rgbloader()` adds horizontal and vertical flips; `valid_rgbloader()` disables augmentation but still samples a random crop. The loader wrappers rely on hard-coded source-size assumptions (`2700×3600` for training and `3600×4800` for validation), rather than reading each dataset's shape before choosing crop coordinates.

The loader uses every filename returned by `os.listdir()` without extension
filtering, so the directory must contain only compatible HDF5 files.

### `GetRAW`

Each HDF5 file must contain:

- `raw_img`: `[H,W]`;
- `raw_color`: `[H,W]`;
- `raw_wb_matrix`: `[4]`;
- `raw_color_matrix`: `[3,3]`.

The RAW wrappers similarly assume `2160×3840` training images and
`2700×4800` validation images when sampling crop origins.
They also use the unfiltered directory listing, so unrelated files cause HDF5
open or missing-dataset errors.

`rgb_generator()` and `raw_generator()` move batches to a requested device and dtype.

## Restoration models

### `FSNET(field_code=False)`

Returns `[quarter, half, full]` predictions. With `field_code=True`, the input
has five channels (RGB plus two image-plane coordinates) and each output is
decoded to RGB. Input dimensions should be divisible by four.

### `RESTORMER(...)`

A four-level transformer encoder-decoder with residual output. Key constructor options are `inp_channels`, `out_channels`, `dim`, `num_blocks`, `num_refinement_blocks`, `heads`, `ffn_expansion_factor`, `bias`, and `LayerNorm_type`. Three downsampling stages generally require dimensions divisible by eight. Because the output is added directly to the input, the current residual path also requires matching input and output channel counts.

### `SRCNN()`

A three-layer convolutional baseline that preserves `[B,3,H,W]`.

### `DocDiff(...)`

With the default four-resolution U-Net, both spatial dimensions should be
divisible by eight so that the three downsampling stages and skip connections
remain shape-compatible.

Training:

```python
init_prediction, noise_prediction, noise_reference = model(gt, degraded)
```

The matching wrapper loss call is:

```python
loss = model.loss(
    gt,
    init_prediction,
    noise_prediction,
    noise_reference,
)
```

Inference:

```python
restored = model.model.inference(degraded)
```

`Schedule(schedule, timesteps)` implements linear and cosine schedules through `get_betas()`. Quadratic and sigmoid schedule functions exist on the class, but `get_betas()` does not currently dispatch them. `GaussianDiffusion` implements residual noising and conditional reverse sampling.

## Internal building blocks

The package exposes implementation-level components from the available
models, including FSNet frequency filters, Restormer attention blocks,
diffusion UNet blocks, Sobel/Laplacian operators, and loss utilities.

Because several subpackages define the same names (`Loss`, `Conv`, `ResBlock`, and `Downsample`), always import internal components from their full module path:

```python
from nets.restormer.model import TransformerBlock
from nets.fsnet.loss import Loss as FSNetLoss
```

## Dataset preparation utilities

| Source helper | Purpose |
| --- | --- |
| `make_rgb.create_h5_dataset(...)` | Converts RGB images to HDF5 files containing `img`. |
| `make_fivek_raw.find_dng_with_glob(root_folder)` | Recursively locates DNG files. |
| `make_raise_raw.find_nef_with_glob(root_folder)` | Recursively locates NEF files. |

These three files are editable preparation scripts, not import-safe library
modules. They contain executable top-level code and placeholder `xxxxxxxxx`
paths; importing `nets.make_rgb`, for example, immediately invokes two
conversions. Replace the source/output paths, ensure output directories exist,
and run the chosen script deliberately. Do not import it merely to access a
helper without first guarding or removing its top-level workflow.

## Small training and image helpers

| Helper | Behavior |
| --- | --- |
| `Adder` | Accumulates finite scalar-like values and returns their arithmetic mean; call `reset()` before reuse. `average()` divides by the accepted count, so do not call it before at least one value has been added. |
| `EMA(beta)` | Updates a moving-average model in place from a current model by zipping their parameter iterators; the two models must have matching parameter order and shapes. |
| `postprocess(img, rgb_range)` | Quantizes/clamps an image to the implemented 8-bit grid expressed in `rgb_range` units. |
| `save_images(img, path, name)` | Clamps to `[0,1]` and saves `<path>/<name>.png`; the destination directory must already exist. |
| `crop_or_pad_tensor(img, target_h, target_w)` | Center-crops or reflection-pads one `[C,H,W]` tensor to the target size. |
