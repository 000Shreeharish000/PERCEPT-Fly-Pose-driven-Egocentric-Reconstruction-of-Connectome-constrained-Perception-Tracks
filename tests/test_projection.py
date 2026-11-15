"""Tests for egocentric projection and ommatidial excitation."""

import numpy as np

from egoflynet.io import PoseData
from egoflynet.body import compute_body_frame
from egoflynet.eye import create_hexagonal_eye_lattice
from egoflynet.projection import project_partner_onto_eyes


def test_projection_partner_straight_ahead():
    # Focal fly at origin looking down +X axis
    # Partner fly directly ahead at (5, 0, 0)
    tracks = np.zeros((1, 2, 5, 3))
    tracks[0, 0, 0] = [1.0, 0.0, 0.0]  # focal head
    tracks[0, 0, 1] = [0.0, 0.0, 0.0]  # focal thorax

    # Partner placed 4 mm directly in front of focal head
    tracks[0, 1, 0] = [5.0, 0.0, 0.0]  # partner head
    tracks[0, 1, 1] = [5.5, 0.0, 0.0]  # partner thorax

    pose = PoseData(tracks=tracks)
    frame = compute_body_frame(pose, focal_fly_idx=0)
    lattice = create_hexagonal_eye_lattice()

    intensities = project_partner_onto_eyes(pose, frame, lattice, dense_body=False)
    assert intensities.shape == (1, lattice.n_ommatidia)

    # Ommatidia pointing near azimuth=0 and elevation=0 should have highest excitation
    az_abs = np.abs(lattice.azimuth)
    el_abs = np.abs(lattice.elevation)
    central_mask = (az_abs < np.deg2rad(15)) & (el_abs < np.deg2rad(15))

    assert np.any(central_mask)
    max_idx = np.argmax(intensities[0])
    assert central_mask[max_idx], "Peak ommatidium excitation must be near center ahead"


def test_projection_partner_left_vs_right():
    # Focal fly pointing along +X
    # Partner placed on the left side (in body coordinates: y < 0 in standard Cartesian left)
    # Note: in head frame, right is +y, left is -y (azimuth < 0 is left eye)
    tracks = np.zeros((1, 2, 5, 3))
    tracks[0, 0, 0] = [1.0, 0.0, 0.0]
    tracks[0, 0, 1] = [0.0, 0.0, 0.0]

    # Partner placed at x=3, y=-3 (to the fly's left)
    tracks[0, 1, 0] = [3.0, -3.0, 0.0]
    tracks[0, 1, 1] = [3.5, -3.0, 0.0]

    pose = PoseData(tracks=tracks)
    frame = compute_body_frame(pose, focal_fly_idx=0)
    lattice = create_hexagonal_eye_lattice()

    intensities = project_partner_onto_eyes(pose, frame, lattice, dense_body=False)

    left_sum = np.sum(intensities[0, lattice.left_indices])
    right_sum = np.sum(intensities[0, lattice.right_indices])

    assert left_sum > right_sum, "Left partner must produce greater excitation in left eye"


def test_projection_distance_attenuation():
    # Partner close (dist = 2 mm) vs Partner far (dist = 8 mm)
    tracks = np.zeros((2, 2, 5, 3))
    # Frame 0: Close
    tracks[0, 0, 0] = [1.0, 0.0, 0.0]
    tracks[0, 0, 1] = [0.0, 0.0, 0.0]
    tracks[0, 1, 0] = [3.0, 0.0, 0.0]

    # Frame 1: Far
    tracks[1, 0, 0] = [1.0, 0.0, 0.0]
    tracks[1, 0, 1] = [0.0, 0.0, 0.0]
    tracks[1, 1, 0] = [9.0, 0.0, 0.0]

    pose = PoseData(tracks=tracks)
    frame = compute_body_frame(pose, focal_fly_idx=0)
    lattice = create_hexagonal_eye_lattice()

    intensities = project_partner_onto_eyes(pose, frame, lattice, dense_body=False)

    close_flux = np.sum(intensities[0])
    far_flux = np.sum(intensities[1])

    assert close_flux > far_flux * 3.0, "Closer partner must produce significantly higher excitation"
