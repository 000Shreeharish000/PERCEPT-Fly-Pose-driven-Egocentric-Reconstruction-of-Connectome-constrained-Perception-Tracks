"""Hexagonal ommatidia compound eye lattice geometry for Drosophila."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class OmmatidiaLattice:
    """Geometric configuration of the fly's compound eyes.

    NOTE: This is an idealized, approximate model of the Drosophila compound eye.
    Biological eyes feature variable interommatidial spacing, eye curvature,
    and pseudopupil characteristics not captured by a uniform lattice.

    Attributes:
        directions: (N, 3) Unit optical axes of ommatidia in head frame:
                    x > 0 is forward, y > 0 is right, z > 0 is up.
        eye_id: (N,) 0 for left eye, 1 for right eye.
        azimuth: (N,) Azimuth angles in radians [-pi, pi].
        elevation: (N,) Elevation angles in radians [-pi/2, pi/2].
        acceptance_angle_rad: Half-width of Gaussian acceptance angle (rad).
    """

    directions: np.ndarray
    eye_id: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    acceptance_angle_rad: float

    @property
    def n_ommatidia(self) -> int:
        return int(len(self.eye_id))

    @property
    def left_indices(self) -> np.ndarray:
        return np.where(self.eye_id == 0)[0]

    @property
    def right_indices(self) -> np.ndarray:
        return np.where(self.eye_id == 1)[0]


def create_hexagonal_eye_lattice(
    n_cols: int = 24,
    n_rows: int = 15,
    azimuth_span_deg: float = 175.0,
    elevation_span_deg: float = 130.0,
    binocular_overlap_deg: float = 20.0,
    acceptance_angle_deg: float = 5.0,
) -> OmmatidiaLattice:
    """Generate a biologically parameterized hexagonal ommatidia lattice for both eyes.

    Each eye covers approximately:
      - Azimuth: from front overlap to lateral/posterior field.
      - Elevation: ventral to dorsal field.
    Hexagonal packing is produced via alternating row offsets.

    Args:
        n_cols: Number of ommatidial columns per eye.
        n_rows: Number of ommatidial rows per eye.
        azimuth_span_deg: Total azimuthal field of view per eye (degrees).
        elevation_span_deg: Total elevation field of view per eye (degrees).
        binocular_overlap_deg: Frontal binocular overlap between eyes (degrees).
        acceptance_angle_deg: Acceptance angle half-width (degrees) for Gaussian blur.

    Returns:
        OmmatidiaLattice instance containing all optical axes and angles.
    """
    acceptance_rad = float(np.deg2rad(acceptance_angle_deg))
    elev_min = -np.deg2rad(elevation_span_deg / 2.0)
    elev_max = np.deg2rad(elevation_span_deg / 2.0)
    elev_steps = np.linspace(elev_min, elev_max, n_rows)

    d_az = np.deg2rad(azimuth_span_deg) / max(n_cols - 1, 1)

    all_dirs: list[np.ndarray] = []
    all_eye_ids: list[int] = []
    all_azimuths: list[float] = []
    all_elevations: list[float] = []

    # Binocular overlap front limit in radians
    overlap_rad = np.deg2rad(binocular_overlap_deg)

    # 1. Left Eye (azimuth negative to slightly positive)
    # Azimuth ranges from -(azimuth_span_deg - overlap_rad) up to +overlap_rad
    az_left_start = -np.deg2rad(azimuth_span_deg) + overlap_rad
    az_left_end = overlap_rad

    for r_idx, el in enumerate(elev_steps):
        # Hexagonal staggering: odd rows are shifted by half a column
        hex_offset = (d_az * 0.5) if (r_idx % 2 == 1) else 0.0
        base_az = np.linspace(az_left_start, az_left_end, n_cols) + hex_offset

        for az in base_az:
            # Spherical to Cartesian: x=forward, y=right, z=up
            # In head frame: x = cos(el)*cos(az), y = cos(el)*sin(az), z = sin(el)
            x = np.cos(el) * np.cos(az)
            y = np.cos(el) * np.sin(az)
            z = np.sin(el)
            vec = np.array([x, y, z], dtype=np.float64)
            vec /= np.linalg.norm(vec)

            all_dirs.append(vec)
            all_eye_ids.append(0)
            all_azimuths.append(az)
            all_elevations.append(el)

    # 2. Right Eye (azimuth slightly negative to positive)
    # Mirror symmetric with left eye across the sagittal plane (y -> -y)
    az_right_start = -overlap_rad
    az_right_end = np.deg2rad(azimuth_span_deg) - overlap_rad

    for r_idx, el in enumerate(elev_steps):
        hex_offset = (d_az * 0.5) if (r_idx % 2 == 1) else 0.0
        base_az = np.linspace(az_right_start, az_right_end, n_cols) + hex_offset

        for az in base_az:
            x = np.cos(el) * np.cos(az)
            y = np.cos(el) * np.sin(az)
            z = np.sin(el)
            vec = np.array([x, y, z], dtype=np.float64)
            vec /= np.linalg.norm(vec)

            all_dirs.append(vec)
            all_eye_ids.append(1)
            all_azimuths.append(az)
            all_elevations.append(el)

    directions = np.array(all_dirs, dtype=np.float64)
    eye_id = np.array(all_eye_ids, dtype=np.int32)
    azimuth = np.array(all_azimuths, dtype=np.float64)
    elevation = np.array(all_elevations, dtype=np.float64)

    return OmmatidiaLattice(
        directions=directions,
        eye_id=eye_id,
        azimuth=azimuth,
        elevation=elevation,
        acceptance_angle_rad=acceptance_rad,
    )

