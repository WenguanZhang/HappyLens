# Optical API Reference

The `lens` package provides structure search, prescription optimization, ray tracing, optical analysis, and tolerancing. The reference is split by responsibility so that data flow and ownership remain clear.

```text
Ray + Surface
      │
      ▼
   System ──► Analysis
      │
      ├─────► Merit / MeritZ / DLS
      ├─────► Delano / Zoom / Generation
      └─────► Deletion / Tolerancing
```

## Reference pages

| Page | Contents |
| --- | --- |
| [System](api/system.md) | Construction, JSON loading, ray sampling, propagation, parameter control, persistence, and multi-configuration data. |
| [Surfaces](api/surfaces.md) | Spherical, even-aspheric, Q-con, Q-bfs, and Binary2 surfaces; object/image boundaries; coordinate transforms; and packages. |
| [Rays and solvers](api/ray_tracing.md) | `Ray`, tensor layouts, pupil sampling, intersection solvers, forward/reverse tracing, and differentiable propagation operators. |
| [Analysis](api/analysis.md) | Layouts, spot diagrams, wavefronts, PSFs, MTF, distortion, illumination, and saved reports. |
| [Optimization](api/optimization.md) | `Merit`, merit operands, population-based and stochastic search, differentiable PSFs, `MeritZ`, and DLS. |
| [Structure search](api/structure_search.md) | Delano generation, zoom generation, generation-specific parameter groups, and component deletion. |
| [Tolerancing](api/tolerancing.md) | Coordinate packages, decenter, tilt, thickness errors, parameter access, and digital compensation workflow. |
| [Utilities](api/utilities.md) | YAML loading, materials, Zernike functions, random seeds, gradient helpers, and Zemax export. |

## Public API levels

- **Primary** interfaces are suitable for normal user code.
- **Advanced** interfaces expose batched optimization and research workflows.
- **Internal** interfaces are implementation details and may change without notice.

The package currently exports symbols through wildcard imports. When names collide, import from the concrete module:

```python
from lens.system import System
from lens.analysis import Analysis
from lens.optim import Merit
```
