# About This Documentation

HappyLens is an open-source, PyTorch-based framework focused on optical optimization and automated lens-structure search. This directory contains the user guide, workflow examples, and API reference for the `lens` and `nets` packages.

The documentation is generated with Sphinx and the Read the Docs theme. Its main sections are:

- **Getting Started** introduces installation, conventions, configuration files, and a first optical analysis.
- **Core Concepts** explains how surfaces, rays, systems, analysis, and optimization interact.
- **Example Gallery** describes the research applications under `test_*` without treating them as stable APIs.
- **API Reference** documents the framework implementation under `lens/` and `nets/`.
- **Project** records licensing and release information.

## Core packages

| Package | Responsibility |
| --- | --- |
| `lens` | Optical surfaces, batched ray tracing, image-quality analysis, merit functions, structure generation, deletion, zoom, and tolerancing. |
| `nets` | RGB/RAW degradation, ISP, dataset loaders, licensed restoration models, diffusion, training losses, and explicit placeholders for implementations that cannot be redistributed. |

For runnable setup and analysis code, use the [Quickstart](quickstart.md). For tensor shapes and units, read [Conventions](conventions.md) before the API reference.

For the licensing and availability status of each network implementation, see
[Third-Party Network Implementations](third_party.md).

## Stability notice

HappyLens 1.0 is the first formal release and the project remains under active
development. The current package-level `__init__.py` files use wildcard
exports. Prefer the documented entry points, review release notes when updating,
and pin a released version or commit when reproducing research results.
