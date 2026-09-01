# Basic Lens Analysis

Source directory: `test_basic/`

## Prime lens example

`test_basic.py` loads a named prescription (currently `gauss`), prints first-order quantities, and constructs both `Analysis` and `Merit` objects.

The notebook-style cells demonstrate:

1. loading YAML with `GetYaml`;
2. selecting the default device and random seed;
3. loading a JSON prescription with `System`;
4. inspecting EFFL, F-number, pupil data, and total track;
5. drawing a layout and spot diagram;
6. computing relative illumination and distortion;
7. computing wavefronts, diffraction PSFs, geometric PSFs, and MTFs;
8. selecting trainable surface parameters;
9. evaluating one combined merit function and optimizer step.

Run from the example directory:

```bash
cd test_basic
python test_basic.py
```

The file is organized as interactive `# %%` cells and is best run cell by
cell. Running the entire file executes every expensive PSF/MTF example and an
optimizer step; the supplied YAML also selects CUDA, and the script constructs
fused Adam. Treat the command above as the required working-directory context,
not as a lightweight CPU smoke test.

Before executing the optimization cell, create `results/` at the repository
root or change the `path='../results'` argument. The plotting helper does not
create that directory itself.

Change the `name` variable near the top to select another compatible pair from
`lens_json/` and `lens_yaml/`. Each YAML selects its material catalogs before
the prescription is constructed. The directly loadable prime-lens pairs are
`f_006`, `g_014`, `g_015`, `gauss`, `l_004`, `l_022`, `phone`, and `wide_50`;
`g_015` merges Schott and Ohara, in that priority order, to supply both its
Schott materials and Ohara `LAH52`. The `phone` configuration uses the bundled
plastic catalog for `APL5014CL` and `POLYSTYR`. The zoom pair `zoom_3x` belongs
with `test_basic_zoom.py`, not the prime-lens script.

## Zoom example

`test_basic_zoom.py` follows the same analysis sequence for `zoom_3x`. It demonstrates how `cfg_id` selects a zoom position and how `MeritZ` replaces `Merit`.

```bash
cd test_basic
python test_basic_zoom.py
```

The same interactive/CUDA caveat applies to the zoom file.
`lens_yaml/zoom_3x.yaml` explicitly sets `VIG: null`, so no additional pupil
clipping is applied. As in the prime example, create the repository-root
`results/` directory before the loss-composition plotting cell.

The script also registers averaged gradients for parameters shared across configurations according to `system.zoom_type`.

## Recommended use

These files are the best starting point for interactive exploration. They use `# %%` cell markers, so they can be opened in editors that support Python cells. For automated testing, extract small deterministic assertions instead of running the complete files.
