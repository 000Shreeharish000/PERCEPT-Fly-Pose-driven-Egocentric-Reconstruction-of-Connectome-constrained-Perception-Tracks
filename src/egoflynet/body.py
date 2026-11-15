"""Focal-fly body coordinate frame reconstruction from pose keypoints."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .io import PoseData


@dataclass
class FlyBodyFrame:
    """Rigid body coordinate frame for the focal fly across frames.

    Attributes:
        origin: (n_frames, 3) 3D coordinate of the head/eye center in world coordinates.
        forward: (n_frames, 3) Unit forward (anteroposterior) vector.
        right: (n_frames, 3) Unit right (mediolateral) vector.
        up: (n_frames, 3) Unit dorsal (dorsoventral) vector.
        yaw: (n_frames,) Heading yaw angle in radians [-pi, pi].
        pitch: (n_frames,) Pitch angle in radians [-pi/2, pi/2].
        roll: (n_frames,) Roll angle in radians [-pi, pi].
        rotations: (n_frames, 3, 3) Rotation matrices where columns are [forward, right, up].
    """

    origin: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    yaw: np.ndarray
    pitch: np.ndarray
    roll: np.ndarray
    rotations: np.ndarray

    def world_to_body(self, world_points: np.ndarray, frame_idx: int) -> np.ndarray:
        """Transform points from world frame to the focal fly's body frame.

        Args:
            world_points: Array of shape (..., 3) in world space.
            frame_idx: Index of the frame to use.

        Returns:
            Array of shape (..., 3) in body coordinates, where:
                x > 0 is forward (anterior)
                y > 0 is right (dextral)
                z > 0 is up (dorsal)
        """
        orig = self.origin[frame_idx]
        rot = self.rotations[frame_idx]  # Shape (3, 3), orthonormal
        # p_ego = R^T * (p_world - origin)
        diff = world_points - orig
        return np.einsum("ij,...j->...i", rot.T, diff)


def compute_body_frame(pose: PoseData, focal_fly_idx: int = 0) -> FlyBodyFrame:
    """Compute orthonormal body coordinate frames from fly keypoints.

    Constructs the forward direction from the thorax-to-head vector, derives
    an orthogonal right vector, and calculates the dorsal up vector.

    Args:
        pose: PoseData object with tracked keypoints.
        focal_fly_idx: Index of the focal fly (0-indexed).

    Returns:
        FlyBodyFrame containing origin and orthonormal basis vectors per frame.
    """
    pose_3d = pose.as_3d()
    n_frames = pose_3d.n_frames

    # Required keypoints
    try:
        head = pose_3d.get_keypoint(focal_fly_idx, "head")
        thorax = pose_3d.get_keypoint(focal_fly_idx, "thorax")
    except KeyError:
        # Fallback to first two keypoints if named ones missing
        head = pose_3d.tracks[:, focal_fly_idx, 0, :]
        thorax = pose_3d.tracks[:, focal_fly_idx, 1, :]

    origin = head.copy()

    # Forward vector: thorax -> head
    fwd_raw = head - thorax
    norm_fwd = np.linalg.norm(fwd_raw, axis=-1, keepdims=True)
    norm_fwd[norm_fwd == 0] = 1.0
    forward = fwd_raw / norm_fwd

    # Handle edge cases (e.g. initial frame where head == thorax)
    for i in range(n_frames):
        if np.allclose(fwd_raw[i], 0):
            forward[i] = forward[i - 1] if i > 0 else np.array([1.0, 0.0, 0.0])

    # Reference world up vector
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Compute right = world_up x forward (so +y is right, +x is forward, +z is up)
    right_raw = np.cross(world_up[np.newaxis, :], forward)
    norm_right = np.linalg.norm(right_raw, axis=-1, keepdims=True)

    # If fly is pitching directly up/down, use fallback vector
    degenerate = (norm_right.squeeze(-1) < 1e-4)
    if np.any(degenerate):
        fallback_ref = np.array([0.0, 1.0, 0.0])
        right_raw[degenerate] = np.cross(fallback_ref[np.newaxis, :], forward[degenerate])
        norm_right = np.linalg.norm(right_raw, axis=-1, keepdims=True)
        norm_right[norm_right == 0] = 1.0

    right = right_raw / norm_right

    # Orthonormal up = forward x right
    up = np.cross(forward, right)
    norm_up = np.linalg.norm(up, axis=-1, keepdims=True)
    norm_up[norm_up == 0] = 1.0
    up = up / norm_up

    # Compute Euler angles (Tait-Bryan: Yaw -> Pitch -> Roll)
    # Yaw: angle of forward vector in xy-plane
    yaw = np.arctan2(forward[:, 1], forward[:, 0])
    # Pitch: elevation of forward vector
    pitch = -np.arcsin(np.clip(forward[:, 2], -1.0, 1.0))
    # Roll: rotation around forward axis
    roll = np.arctan2(right[:, 2], up[:, 2])

    # Rotation matrix per frame: columns are [forward, right, up]
    rotations = np.stack([forward, right, up], axis=-1)

    return FlyBodyFrame(
        origin=origin,
        forward=forward,
        right=right,
        up=up,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        rotations=rotations,
    )
