"""Synthetic multi-fly pose trajectory generation for Drosophila courtship behavior."""

from __future__ import annotations

import numpy as np

from .io import PoseData, STANDARD_KEYPOINTS


def generate_courtship_trajectory(
    n_frames: int = 180,
    fps: float = 60.0,
    seed: int = 42,
    arena_radius_mm: float = 20.0,
) -> PoseData:
    """Generate synthetic 2-fly pose tracking data mimicking Drosophila courtship.

    Fly 0 (focal/chaser fly, e.g. male) tracks and orients toward Fly 1 (partner/female).
    Includes realistic morphometrics, heading updates, and unilateral wing extension.

    Args:
        n_frames: Total number of frames to generate.
        fps: Recording frame rate (Hz).
        seed: Random seed for reproducibility.
        arena_radius_mm: Approximate arena size in millimeters.

    Returns:
        PoseData instance with shape (n_frames, 2, 5, 3) in millimeters.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    t = np.arange(n_frames) * dt

    # Keypoint names: ["head", "thorax", "abdomen_tip", "wing_left", "wing_right"]
    n_flies = 2
    n_kps = len(STANDARD_KEYPOINTS)
    tracks = np.zeros((n_frames, n_flies, n_kps, 3), dtype=np.float64)

    # Drosophila melanogaster typical dimensions (in mm)
    head_dist = 0.55
    abdomen_dist = 1.30
    wing_length = 1.60
    wing_rest_angle = np.deg2rad(15.0)

    # 1. Partner (Female) Trajectory: walking smoothly in an arc/circle
    partner_speed = 8.0  # mm/s
    partner_ang_vel = 0.6  # rad/s
    partner_r = 10.0  # mm
    partner_phase = rng.uniform(0, 2 * np.pi)

    female_x = partner_r * np.cos(partner_ang_vel * t + partner_phase)
    female_y = partner_r * np.sin(partner_ang_vel * t + partner_phase)
    female_z = 0.3 + 0.05 * np.sin(4.0 * np.pi * t)  # subtle walking oscillation

    # Female heading (tangent to circle with slight noise)
    female_yaw = partner_ang_vel * t + partner_phase + np.pi / 2.0

    for i in range(n_frames):
        yaw = female_yaw[i]
        pos = np.array([female_x[i], female_y[i], female_z[i]])
        fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        tracks[i, 1, 1] = pos  # thorax
        tracks[i, 1, 0] = pos + head_dist * fwd  # head
        tracks[i, 1, 2] = pos - abdomen_dist * fwd  # abdomen_tip

        # Wings at rest along abdomen
        w_left_dir = -np.cos(wing_rest_angle) * fwd - np.sin(wing_rest_angle) * right
        w_right_dir = -np.cos(wing_rest_angle) * fwd + np.sin(wing_rest_angle) * right
        tracks[i, 1, 3] = pos + wing_length * w_left_dir + 0.1 * up  # wing_left
        tracks[i, 1, 4] = pos + wing_length * w_right_dir + 0.1 * up  # wing_right

    # 2. Focal (Male) Trajectory: pursues female from behind, then closes in
    male_pos = np.array([female_x[0] - 5.0, female_y[0] - 3.0, 0.3])
    male_yaw = female_yaw[0]

    for i in range(n_frames):
        target_pos = np.array([female_x[i], female_y[i], female_z[i]])

        # Vector to female
        to_female = target_pos - male_pos
        dist_to_female = np.linalg.norm(to_female[:2])

        # Male moves toward female, aiming to maintain ~2.5 mm courtship distance
        desired_yaw = np.arctan2(to_female[1], to_female[0])
        yaw_err = (desired_yaw - male_yaw + np.pi) % (2 * np.pi) - np.pi
        male_yaw += np.clip(yaw_err, -4.0 * dt, 4.0 * dt)

        # Chase speed depends on distance
        approach_speed = np.clip((dist_to_female - 2.5) * 4.0, 0.0, 15.0)
        fwd = np.array([np.cos(male_yaw), np.sin(male_yaw), 0.0])
        right = np.array([-np.sin(male_yaw), np.cos(male_yaw), 0.0])
        up = np.array([0.0, 0.0, 1.0])

        male_pos[:2] += fwd[:2] * approach_speed * dt
        male_pos[2] = 0.3 + 0.05 * np.cos(4.0 * np.pi * t[i])

        tracks[i, 0, 1] = male_pos  # thorax
        tracks[i, 0, 0] = male_pos + head_dist * fwd  # head
        tracks[i, 0, 2] = male_pos - abdomen_dist * fwd  # abdomen_tip

        # Courtship wing extension: left wing opens between frames 40 and 120
        is_singing = 40 <= i <= 130
        left_angle = np.deg2rad(70.0) if is_singing else wing_rest_angle
        right_angle = wing_rest_angle

        w_left_dir = -np.cos(left_angle) * fwd - np.sin(left_angle) * right
        w_right_dir = -np.cos(right_angle) * fwd + np.sin(right_angle) * right

        tracks[i, 0, 3] = male_pos + wing_length * w_left_dir + 0.1 * up
        tracks[i, 0, 4] = male_pos + wing_length * w_right_dir + 0.1 * up

    return PoseData(
        tracks=tracks,
        keypoint_names=list(STANDARD_KEYPOINTS),
        fps=fps,
        timestamps=t,
    )

