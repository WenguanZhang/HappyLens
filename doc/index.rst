HappyLens Documentation
=======================

.. note::

   HappyLens 1.0 is the first formal release. The project remains under active
   development; use the documented entry points and review release notes when
   updating between versions.

**HappyLens** is an open-source, PyTorch-based framework for lens design, with
a particular focus on optical optimization and automated structure search. It
supports first-order optimization, parallel population-based optimization,
heuristic and gradient-based optimizers, and optical systems with multiple
configurations.

With HappyLens, you can:

* search prime and zoom structures without requiring a patent or lens library
  as an initialization source;
* trace and optimize candidate populations on the GPU;
* use simulated annealing, genetic algorithms, differential evolution,
  gradient-based optimizers, and damped least squares;
* model prime lenses and systems with multiple configurations, including zoom
  and internal-focusing lenses;
* use spherical, even-aspheric, Q-con, Q-bfs, and Binary2 surfaces;
* evaluate geometric image quality and diffraction-aware PSFs, including
  differentiable optics--ISP--restoration workflows.

.. code-block:: python

   import torch
   import lens

   torch.set_default_dtype(torch.float64)
   device = "cuda:0" if torch.cuda.is_available() else "cpu"
   torch.set_default_device(device)

   cfg = lens.GetYaml("lens_yaml/gauss.yaml")
   lens.configure_material_catalog(cfg.MATERIAL_CATALOG)
   system = lens.System(
       wavelengths=cfg.WAVELENGTHS,
       waveweights=cfg.WAVEWEIGHTS,
       p_wvl=cfg.P_WAVE,
       max_view=cfg.MAX_VIEW,
       norm_views=cfg.NORM_VIEWS,
       azimuths=cfg.AZIMUTHS,
       file="lens_json/gauss.json",
   )

   analysis = lens.Analysis(system)
   analysis.plot_setup_with_trace()

Getting Started
---------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   start_here
   README
   quickstart
   conventions
   configuration

Core Concepts
-------------

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   architecture

Example Gallery
---------------

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   gallery/introduction
   gallery/basic
   gallery/prime_generation
   gallery/zoom_generation
   gallery/component_deletion
   gallery/catadioptric

API Reference
-------------

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   lens_api
   api/system
   api/surfaces
   api/ray_tracing
   api/analysis
   api/optimization
   api/structure_search
   api/tolerancing
   api/utilities
   nets_api

Project
-------

.. toctree::
   :maxdepth: 1
   :caption: Project

   license
   third_party
