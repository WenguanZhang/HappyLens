# Third-Party Network Implementations

HappyLens-authored code is distributed under Apache License 2.0. Some network
implementations under `nets/` are adaptations of third-party research code and
retain their upstream licenses. The repository-level Apache license does not
replace those upstream terms.

## Bundled implementations

| HappyLens module | Upstream project | License | Status |
| --- | --- | --- | --- |
| `nets/restormer` | [Restormer](https://github.com/swz30/Restormer) | MIT | Bundled |
| `nets/fsnet` | [FSNet](https://github.com/c-yn/FSNet) | MIT | Bundled |
| `nets/diff` | [DocDiff](https://github.com/Royalvice/DocDiff) | MIT | Bundled |
| `nets/srcnn` | SRCNN paper | HappyLens Apache-2.0 | Independently implemented and bundled |

The complete upstream MIT texts are stored in `third_party_licenses/`. Source
attribution and the adaptation boundary are recorded in
`THIRD_PARTY_NOTICES.md` at the repository root.

## Implementations not distributed

The following upstream repositories did not provide an explicit
redistribution license when reviewed. HappyLens therefore does not bundle the
adapted implementations:

| Reserved model name | Upstream project | Local placeholder |
| --- | --- | --- |
| `MIMOUNET`, `MIMOUNET+F` | [MIMO-UNet](https://github.com/chosj95/mimo-unet) | `nets/mimounet` |
| `DWDN` | [DWDN](https://github.com/dongjxjx/dwdn) | `nets/dwdn` |
| `CDWDN` | Derived from DWDN | `nets/cdwdn` |
| `DEEPSN`, `DEEPSN+F` | [DeepSN-Net](https://github.com/pandazcx/DeepSN-Net) | `nets/deepsn` |

Each placeholder directory contains only `model.py` and `loss.py`. Instantiating
one of their classes, or selecting the corresponding name through
`nets.Model`, raises a `RuntimeError` explaining that the implementation is not
distributed and providing the upstream URL.

```python
import nets

# Raises RuntimeError with the DWDN upstream URL.
model = nets.Model("DWDN")
```

The placeholder is intentional: attribution, a citation, or a research-only
notice does not substitute for permission to redistribute source code.

## Model weights and datasets

Source-code licenses do not automatically cover pretrained weights or
datasets. HappyLens does not grant rights to third-party checkpoints or data;
review the terms published by their respective providers before downloading,
using, or redistributing them.
