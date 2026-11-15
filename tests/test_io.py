"""Tests for I/O and PoseData validation."""

from pathlib import Path
import numpy as np
import pytest

from egoflynet.io import PoseData, load_csv, save_csv, STANDARD_KEYPOINTS


def test_pose_data_validation():
    # Valid 2-fly, 5-keypoint 2D data
    tracks = np.zeros((10, 2, 5, 2))
    pose = PoseData(tracks=tracks)
    assert pose.n_frames == 10
    assert pose.n_flies == 2
    assert pose.n_keypoints == 5
    assert not pose.is_3d

    # Promote to 3D
    pose_3d = pose.as_3d()
    assert pose_3d.is_3d
    assert pose_3d.n_coords == 3

    # Invalid rank
    with pytest.raises(ValueError, match="rank 4"):
        PoseData(tracks=np.zeros((10, 2, 5)))

    # Empty frames
    with pytest.raises(ValueError, match="0 frames"):
        PoseData(tracks=np.zeros((0, 2, 5, 2)))

    # Keypoint name count mismatch
    with pytest.raises(ValueError, match="Mismatch"):
        PoseData(tracks=tracks, keypoint_names=["head", "thorax"])


def test_get_keypoint():
    tracks = np.zeros((5, 2, 5, 3))
    tracks[:, 0, 0, :] = 1.23  # fly 0, head
    pose = PoseData(tracks=tracks)

    head = pose.get_keypoint(0, "head")
    assert head.shape == (5, 3)
    assert np.allclose(head, 1.23)

    with pytest.raises(IndexError):
        pose.get_keypoint(2, "head")

    with pytest.raises(KeyError):
        pose.get_keypoint(0, "nonexistent_limb")


def test_csv_save_and_load(tmp_path: Path):
    tracks = np.random.default_rng(0).normal(size=(8, 2, 5, 3))
    orig_pose = PoseData(tracks=tracks, fps=30.0)

    csv_path = tmp_path / "test_pose.csv"
    save_csv(orig_pose, csv_path)
    assert csv_path.exists()

    loaded_pose = load_csv(csv_path, fps=30.0)
    assert loaded_pose.n_frames == 8
    assert loaded_pose.n_flies == 2
    assert loaded_pose.n_keypoints == 5
    assert loaded_pose.is_3d
    assert np.allclose(orig_pose.tracks, loaded_pose.tracks, atol=1e-3)
