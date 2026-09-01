# Quickstart — Your First 5 Minutes

## 1. Prepare the environment

HappyLens uses Python 3.12 and is not yet distributed as an installable Python package. Install a PyTorch build compatible with your CUDA environment, then install the repository dependencies and run Python from the repository root.

Commands in this documentation use `python` to mean the selected Python 3.12
interpreter. On systems where that command is not installed (commonly macOS),
use `python3` instead; on Windows, `py -3.12` is another option.

```bash
cd /path/to/HappyLens
python -m pip install -r requirements.txt
python
```

The root requirements file includes dependencies used by the framework and the research examples. Documentation-only dependencies remain in `doc/requirements.txt`.

## 2. Load a lens

```python
import torch
import lens

torch.set_default_dtype(torch.float64)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

cfg = lens.GetYaml("lens_yaml/gauss.yaml")
lens.configure_material_catalog(cfg.MATERIAL_CATALOG)
lens.set_random_seed(cfg.SEED)

system = lens.System(
    wavelengths=cfg.WAVELENGTHS,
    waveweights=cfg.WAVEWEIGHTS,
    p_wvl=cfg.P_WAVE,
    max_view=cfg.MAX_VIEW,
    sys_num=cfg.SYS_NUM,
    cfg_num=cfg.CFG_NUM,
    pre_samp=cfg.PRE_SAMP,
    fix_radius_surf=cfg.FIX_RADIUS_SURF,
    norm_views=cfg.NORM_VIEWS,
    azimuths=cfg.AZIMUTHS,
    vig=cfg.VIG,
    file="lens_json/gauss.json",
)

print("EFFL:", system.EFFL)
print("F/#:", system.FNO)
print("Total track:", system.TOTR)
```

## 3. Trace rays

```python
rays = system.sample_ray_2d(
    sampling=31,
    norm_view=[0.0, 0.7, 1.0],
    azimuth=0.0,
)
rays = system.propagate(rays)

print(rays.o.shape)
print(rays.valid.float().mean())
```

## 4. Analyze image quality

```python
analysis = lens.Analysis(system)
analysis.plot_setup_with_trace(sys_id=0, cfg_id=0)
analysis.spot_diagram(sys_id=0, cfg_id=0, sampling=31)
```

```python
psf = analysis.psf(
    sys_id=0,
    cfg_id=0,
    pupil_samp=127,
    image_samp=63,
    image_delta=0.5,
    norm_view=1.0,
    azimuth=0.0,
    show=False,
)

freq, tangential, sagittal = analysis.mtf(
    sys_id=0,
    cfg_id=0,
    pupil_samp=127,
    image_samp=63,
    image_delta=0.5,
    norm_view=1.0,
    azimuth=0.0,
    freq_max=500,
    show=False,
)
```

## 5. Compute a differentiable merit function

```python
merit = lens.Merit(system, samp_rays=cfg.SAMP_RAYS)
optimizer = torch.optim.Adam(merit.params_lr(cfg.LR_OPT))

optimizer.zero_grad()
loss_per_system = merit.forward_loss(cfg.MERIT)
loss = loss_per_system[system.valid].sum()
loss.backward()
optimizer.step()

merit.update_system(rmax=cfg.MAX_RADIUS, avg_cfg=True)
```

`Analysis` is intended primarily for gradient-free reporting and visualization. Use the differentiable functions in `Merit` when optimizing optical parameters jointly with an imaging pipeline.

`System.update()` only recomputes paraxial quantities such as EFFL, pupils,
F-number, and total track. `Merit.update_system()` performs the broader
post-step maintenance used above: optional material fitting, aperture update,
quick focus, and repeated paraxial refreshes.
