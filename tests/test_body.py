"""Tests for focal fly body coordinate frame construction."""

import numpy as np
import pytest

from egoflynet.io import PoseData
from egoflynet.body import compute_body_frame


def test_body_frame_forward_x():
    # Fly 0: Thorax at (0, 0, 0), Head at (1, 0, 0) -> pointing directly +X
    tracks = np.zeros((1, 1, 5, 3))
    tracks[0, 0, 0] = [1.0, 0.0, 0.0]  # head
    tracks[0, 0, 1] = [0.0, 0.0, 0.0]  # thorax

    pose = PoseData(tracks=tracks)
    frame = compute_body_frame(pose, focal_fly_idx=0)

    # Forward should be [1, 0, 0]
    assert np.allclose(frame.forward[0], [1.0, 0.0, 0.0], atol=1e-5)
    # Right should be [0, 1, 0]
    assert np.allclose(frame.right[0], [0.0, 1.0, 0.0], atol=1e-5)
    # Up should be [0, 0, 1]
    assert np.allclose(frame.up[0], [0.0, 0.0, 1.0], atol=1e-5)
    # Yaw should be 0
    assert np.isclose(frame.yaw[0], 0.0, atol=1e-5)

    # World to body transform: point at (5, 0, 0) world is (4, 0, 0) relative to head at (1, 0, 0)
    world_target = np.array([5.0, 0.0, 0.0])
    body_target = frame.world_to_body(world_target, frame_idx=0)
    assert np.allclose(body_target, [4.0, 0.0, 0.0], atol=1e-5)


def test_body_frame_forward_y():
    # Fly 0: Thorax at (0, 0, 0), Head at (0, 2, 0) -> pointing directly +Y
    tracks = np.zeros((1, 1, 5, 3))
    tracks[0, 0, 0] = [0.0, 2.0, 0.0]  # head
    tracks[0, 0, 1] = [0.0, 0.0, 0.0]  # thorax

    pose = PoseData(tracks=tracks)
    frame = compute_body_frame(pose, focal_fly_idx=0)

    # Forward should be [0, 1, 0]
    assert np.allclose(frame.forward[0], [0.0, 1.0, 0.0], atol=1e-5)
    # Right should be [-1, 0, 0]
    assert np.allclose(frame.right[0], [-1.0, 0.0, 0.0], atol=1e-5)
    # Up should be [0, 0, 1]
    assert np.allclose(frame.up[0], [0.0, 0.0, 1.0], atol=1e-5)
    # Yaw should be pi/2
    assert np.isclose(frame.yaw[0], np.pi / 2.0, atol=1e-5)
