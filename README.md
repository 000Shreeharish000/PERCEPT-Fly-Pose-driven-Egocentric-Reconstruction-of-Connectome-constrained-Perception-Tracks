# EgoFlyNet (PERCEPT-Fly)

**Pose-driven egocentric visual reconstruction for *Drosophila* with connectome-constrained perception.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

EgoFlyNet reconstructs the dynamic egocentric visual input experienced by a focal fruit fly (*Drosophila melanogaster*) from multi-animal pose-tracking data (e.g. from SLEAP or DeepLabCut) and projects it onto a biologically parameterized hexagonal compound eye lattice (~720 ommatidia). The resulting visual excitation tracks can optionally interface with connectome-constrained neural network models (e.g. [flyvis](https://github.com/turagalab/flyvis), Nature 2024).

---

## Pipeline Overview

```
                      Multi-Animal Pose Data
               (SLEAP HDF5 / CSV / Synthetic Trajectory)
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Focal Fly Body Frame Math           │
             │   - Anteroposterior axis (Thorax → Head)  │
             │   - Orthonormal Right & Dorsal vectors    │
             │   - Local egocentric origin & orientation │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │      Hexagonal Compound Eye Lattice       │
             │   - Biometric coverage (Left & Right eye) │
             │   - Hexagonal staggered rows (720 units)  │
             │   - Optical axes in spherical coordinates │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
             ┌───────────────────────────────────────────┐
             │       Egocentric Visual Projection        │
             │   - World-to-head coordinate transform    │
             │   - Partner limb & segment interpolation  │
             │   - Gaussian receptive field acceptance   │
             │   - Inverse-square solid angle falloff    │
             └─────────────────────┬─────────────────────┘
                                   │
                                   ▼
               (n_frames × n_ommatidia) Excitation Array
                                   │
            ┌──────────────────────┴──────────────────────┐
            ▼                                             ▼
┌───────────────────────────────┐           ┌───────────────────────────────┐
│     Visualization Engine      │           │   Connectome Bridge (flyvis)  │
│  - Top-down arena view        │           │  - Optional lazy integration  │
│  - Egocentric hex-grid view   │           │  - Connectome T4/T5 response  │
│  - Temporal flux time trace   │           │  - Direction-selective motion │
└───────────────────────────────┘           └───────────────────────────────┘
```

---

## Quick Start

### Installation

```sh
# Clone and enter directory
cd egoflynet

# Base installation
pip install -e .

# With optional development and testing tools
pip install -e ".[dev]"

# With optional connectome model support
pip install -e ".[flyvis]"
```

### Running from the Command Line

To run on the built-in synthetic courtship trajectory (requires zero downloads):

```sh
python -m egoflynet run --input synthetic --out outputs --frames 120
```

To run on your own pose-tracking file:

```sh
python -m egoflynet run --input path/to/pose_tracks.csv --out outputs
```

### Python API Usage

```python
from egoflynet import (
    generate_courtship_trajectory,
    compute_body_frame,
    create_hexagonal_eye_lattice,
    project_partner_onto_eyes,
)

# 1. Load or generate tracking data
pose = generate_courtship_trajectory(n_frames=120, fps=60.0)

# 2. Derive focal fly 3D body orientation
body_frame = compute_body_frame(pose, focal_fly_idx=0)

# 3. Build compound eye hexagonal lattice (~720 ommatidia)
lattice = create_hexagonal_eye_lattice()

# 4. Project partner fly onto ommatidia
intensities = project_partner_onto_eyes(pose, body_frame, lattice)
print(f"Excitation shape: {intensities.shape}")  # (120, 720)
```

---

## Research Status and Limitations

In accordance with transparent research practices:

1. **Simplified Body Model:** The fly's orientation is derived from skeletal keypoint vectors (e.g. thorax-to-head) and Gram-Schmidt orthonormalization. It does not account for flexible head-neck yaw pitch adjustments or fine antennal saccades.
2. **Approximate Eye Geometry:** Compound eye ommatidia are modeled on an idealized spherical-patch hexagonal lattice. Real *Drosophila* compound eyes exhibit non-uniform interommatidial spacing, distinct dorsal/anterior acute zones (fovea-like specializations), and individual ommatidial facet tilt angles.
3. **Ray & Occlusion Approximations:** Partner fly geometry is modeled via interpolated point segments. Full ray-traced physical rendering with mutual occlusion and cuticle reflectance properties is not simulated.
4. **Modeled, Not Empirical:** Reconstructed ommatidial excitation is a computational estimate based on forward geometric optics. It is **not** a direct electrophysiological recording from photoreceptors.
5. **No Claims About Subjective Perception:** These outputs model retinal excitation patterns for computational vision and connectome modeling. They do not constitute evidence or claims regarding biological consciousness or intentionality.

---

## Academic References

- **flyvis:** Lappalainen, J. K., Tschopp, F., Grover, D., et al. (2024). *Connectome-constrained networks predict neural activity across the Drosophila visual system*. Nature, 634, 1132–1140. [https://github.com/turagalab/flyvis](https://github.com/turagalab/flyvis) (MIT License)
- **FlyBody:** Vaxenburg, R., et al. (2024). *FlyBody: Realistic biomimetic models of Drosophila melanogaster*. (Apache 2.0)
- **SLEAP:** Pereira, T. D., et al. (2022). *SLEAP: A deep learning system for multi-animal pose tracking*. Nature Methods, 19, 486–495. [https://github.com/talmo/sleap](https://github.com/talmo/sleap) (BSD-3-Clause)

---

## License

MIT License. See [LICENSE](LICENSE) for details.

