# Example Gallery

The `test_*` directories are research workflow scripts built on HappyLens.
They show how framework components are assembled, but they are not lightweight
unit tests or part of the stable API. Some are directly runnable after
installing dependencies; others require CUDA, external datasets, a previously
generated prescription, or documented path/configuration adaptation. Check the
workflow page before starting a long run.

## Gallery map

| Directory | Application |
| --- | --- |
| `test_basic` | Loading prescriptions, tracing, analysis, and one optimization step. |
| `test_prime` | Automatic prime lens structure generation. |
| `test_zoom` | Automatic zoom-lens structure generation and trajectory optimization. |
| `test_del` | Lens-component deletion and RAW computational imaging. |
| `test_cake` | Micro catadioptric design, nested tolerancing, and digital compensation. |

## Shared execution assumptions

Most scripts assume that the current working directory is their own `test_*` directory because paths are written as `./lens_yaml`, `./lens_json`, or `../lens_json`.
The commands below write `python` for the selected Python 3.12 interpreter;
substitute `python3` or `py -3.12` when that is the command provided by your
platform or environment.

```bash
cd test_zoom
python test_gen_zoom.py --name gen_zoom_2x
```

Common behavior:

- optical computation uses `torch.float64`;
- neural networks use `torch.float32`;
- YAML files currently default to `cuda:0`;
- outputs are written below a local `results/` directory;
- TensorBoard logs are stored in each timestamped result directory;
- scripts may run for many iterations and consume substantial GPU memory;
- RAW/RGB examples expect external datasets that are not bundled with the repository.

## Result structure

Depending on the application, a run may produce:

- a copied YAML configuration;
- TensorBoard event files;
- intermediate `epoch_*` directories;
- saved lens JSON prescriptions;
- Zemax text prescriptions;
- layout and spot-diagram SVG files;
- restored or simulated images;
- model checkpoints and quality metrics.

For reproducibility, retain the copied configuration, random seed, source commit, PyTorch/CUDA versions, and dataset version with every reported result.
