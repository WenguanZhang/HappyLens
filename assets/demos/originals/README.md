# Original demo assets

This directory contains the full-resolution sources for the catadioptric,
component-deletion, and optical-digital joint-fine-tuning demos. The
full-resolution prime- and Zoom-lens workflow GIFs live in
`website/assets/workflows` so the standalone website can deploy them directly
without keeping a second copy. `spot-diagram-and-mtf.png` is stored only in the
parent demo directory because its README version requires no optimization.

The optimized README GIFs in the parent directory retain every animation frame
and the original playback timing. Run
`python3.12 tools/compress_demo_assets.py` from the repository root to regenerate
them without overwriting the full-resolution sources.
