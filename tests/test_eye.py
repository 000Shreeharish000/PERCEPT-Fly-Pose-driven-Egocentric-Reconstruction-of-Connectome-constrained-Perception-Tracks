"""Tests for compound eye hexagonal lattice geometry."""

import numpy as np

from egoflynet.eye import create_hexagonal_eye_lattice


def test_lattice_geometry():
    lattice = create_hexagonal_eye_lattice(n_cols=16, n_rows=10)

    # 1. Direction vectors must all be unit length
    norms = np.linalg.norm(lattice.directions, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)

    # 2. Both eyes should have equal ommatidia counts
    assert len(lattice.left_indices) == len(lattice.right_indices)
    assert lattice.n_ommatidia == len(lattice.left_indices) + len(lattice.right_indices)

    # 3. Left eye mostly negative azimuth, right eye mostly positive
    left_az = lattice.azimuth[lattice.left_indices]
    right_az = lattice.azimuth[lattice.right_indices]
    assert np.mean(left_az) < 0
    assert np.mean(right_az) > 0

    # 4. Check elevation limits
    assert np.all(lattice.elevation >= -np.pi / 2.0)
    assert np.all(lattice.elevation <= np.pi / 2.0)
