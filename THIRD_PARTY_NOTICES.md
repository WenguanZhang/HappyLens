# Third-Party Notices

HappyLens contains code adapted from third-party research implementations.
The Apache License 2.0 at the repository root applies to HappyLens-authored
code, while the components below retain their upstream licenses.

## Restormer

- Source: https://github.com/swz30/Restormer
- Local module: `nets/restormer`
- License: MIT
- License copy: `third_party_licenses/Restormer-MIT.txt`
- Modifications: adapted to the HappyLens model and training interfaces.

## FSNet

- Source: https://github.com/c-yn/FSNet
- Local module: `nets/fsnet`
- License: MIT
- License copy: `third_party_licenses/FSNet-MIT.txt`
- Modifications: adapted to the HappyLens model and training interfaces.

## DocDiff

- Source: https://github.com/Royalvice/DocDiff
- Local module: `nets/diff`
- License: MIT
- License copy: `third_party_licenses/DocDiff-MIT.txt`
- Modifications: adapted to the HappyLens deblurring and training interfaces.

## SRCNN

The implementation in `nets/srcnn` was independently written from the
architecture described in the SRCNN paper. No source code from an external
SRCNN implementation is redistributed.

## Unavailable implementations

HappyLens retains API placeholders, but does not distribute adapted
implementations of the following projects because their upstream repositories
did not provide an explicit redistribution license when reviewed:

- MIMO-UNet: https://github.com/chosj95/mimo-unet
- DWDN and the DWDN-derived CDWDN: https://github.com/dongjxjx/dwdn
- DeepSN-Net: https://github.com/pandazcx/DeepSN-Net

The corresponding `model.py` and `loss.py` files only raise an explanatory
runtime error and contain no upstream implementation.
