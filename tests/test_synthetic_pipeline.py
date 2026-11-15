"""End-to-end integration test of synthetic courtship pipeline."""

import numpy as np

from egoflynet.synthetic import generate_courtship_trajectory
from egoflynet.body import compute_body_frame
from egoflynet.eye import create_hexagonal_eye_lattice
from egoflynet.projection import project_partner_onto_eyes
from egoflynet.flyvis_bridge import is_flyvis_available, simulate_connectome_vision
import pytest


def test_synthetic_end_to_end():
    # 1. Generate 30 frames
    pose = generate_courtship_trajectory(n_frames=30, fps=60.0, seed=123)
    assert pose.n_frames == 30
    assert pose.n_flies == 2
    assert pose.is_3d

    # 2. Compute body frame
    body_frame = compute_body_frame(pose, focal_fly_idx=0)
    assert body_frame.forward.shape == (30, 3)
    assert body_frame.origin.shape == (30, 3)

    # 3. Create eye lattice
    lattice = create_hexagonal_eye_lattice(n_cols=18, n_rows=12)
    assert lattice.n_ommatidia > 0

    # 4. Project
    intensities = project_partner_onto_eyes(pose, body_frame, lattice, dense_body=True)
    assert intensities.shape == (30, lattice.n_ommatidia)
    assert np.all(intensities >= 0.0)
    assert np.max(intensities) > 0.0, "Visual excitation must be non-zero during courtship tracking"


def test_flyvis_bridge_guard():
    if not is_flyvis_available():
        with pytest.raises(ImportError, match="flyvis"):
            dummy_signal = np.zeros((10, 100), dtype=np.float32)
            simulate_connectome_vision(dummy_signal)
    else:
        # If installed, run light verification
        pass

