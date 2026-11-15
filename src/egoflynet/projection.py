"""Egocentric visual projection of partner fly onto the hexagonal ommatidia lattice."""

from __future__ import annotations

import numpy as np

from .io import PoseData
from .body import FlyBodyFrame
from .eye import OmmatidiaLattice


def expand_partner_body_points(
    partner_keypoints: np.ndarray,
    keypoint_names: list[str],
    samples_per_segment: int = 5,
) -> np.ndarray:
    """Generate dense geometric point cloud along the partner fly's anatomical axes.

    Interpolates points along:
      - Head to thorax
      - Thorax to abdomen tip
      - Thorax to left wing tip
      - Thorax to right wing tip

    Args:
        partner_keypoints: (n_kps, 3) 3D coordinates of partner.
        keypoint_names: Names matching keypoint indices.
        samples_per_segment: Number of sample points per limb/segment.

    Returns:
        (M, 3) array of 3D points representing the partner fly's body.
    """
    points: list[np.ndarray] = [partner_keypoints]

    def get_kp(name: str) -> np.ndarray | None:
        if name in keypoint_names:
            return partner_keypoints[keypoint_names.index(name)]
        return None

    head = get_kp("head")
    thorax = get_kp("thorax")
    abdomen = get_kp("abdomen_tip")
    wing_l = get_kp("wing_left")
    wing_r = get_kp("wing_right")

    alphas = np.linspace(0.1, 0.9, samples_per_segment)[:, np.newaxis]

    if head is not None and thorax is not None:
        points.append(thorax + alphas * (head - thorax))
    if thorax is not None and abdomen is not None:
        points.append(thorax + alphas * (abdomen - thorax))
    if thorax is not None and wing_l is not None:
        points.append(thorax + alphas * (wing_l - thorax))
    if thorax is not None and wing_r is not None:
        points.append(thorax + alphas * (wing_r - thorax))

    return np.concatenate(points, axis=0)


def project_partner_onto_eyes(
    pose: PoseData,
    body_frame: FlyBodyFrame,
    lattice: OmmatidiaLattice,
    focal_idx: int = 0,
    partner_idx: int = 1,
    min_distance_mm: float = 0.5,
    dense_body: bool = True,
) -> np.ndarray:
    """Project partner fly into egocentric spherical space and rasterize onto ommatidia.

    For each frame, transforms partner keypoints/body points into the focal fly's
    head frame, evaluates the angular offset against each ommatidium's optical axis,
    and applies a Gaussian acceptance function scaled by 1/distance^2.

    Args:
        pose: Pose tracking data.
        body_frame: Reconstructed focal fly body frame.
        lattice: Ommatidia lattice geometry.
        focal_idx: Index of the viewing fly (default 0).
        partner_idx: Index of the viewed partner fly (default 1).
        min_distance_mm: Distance clipping threshold to avoid numerical explosion.
        dense_body: If True, interpolates points across the partner's anatomical segments.

    Returns:
        Array of shape (n_frames, n_ommatidia) with values representing visual excitation.
    """
    pose_3d = pose.as_3d()
    n_frames = pose_3d.n_frames
    n_ommatidia = lattice.n_ommatidia
    axes = lattice.directions  # (n_ommatidia, 3)
    sigma = lattice.acceptance_angle_rad
    sigma_sq2 = 2.0 * (sigma ** 2)

    intensities = np.zeros((n_frames, n_ommatidia), dtype=np.float32)

    for fr in range(n_frames):
        # 1. Partner world coordinates
        partner_kps = pose_3d.tracks[fr, partner_idx]  # (n_kps, 3)
        if np.all(np.isnan(partner_kps)):
            continue

        if dense_body:
            pts_world = expand_partner_body_points(partner_kps, pose_3d.keypoint_names)
        else:
            pts_world = partner_kps

        # 2. Transform into focal fly's egocentric body frame
        pts_ego = body_frame.world_to_body(pts_world, fr)  # (M, 3)

        # 3. Compute distance and unit line-of-sight directions
        dists = np.linalg.norm(pts_ego, axis=-1)  # (M,)
        valid = dists > 1e-4
        if not np.any(valid):
            continue

        pts_ego = pts_ego[valid]
        dists = dists[valid]

        unit_dirs = pts_ego / dists[:, np.newaxis]  # (M, 3)

        # 4. Dot product with all ommatidial axes: (M, 3) @ (3, N) -> (M, N)
        cos_angles = np.dot(unit_dirs, axes.T)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles_rad = np.arccos(cos_angles)  # (M, N)

        # 5. Gaussian receptive field weighting
        rf_weights = np.exp(-(angles_rad ** 2) / sigma_sq2)  # (M, N)

        # 6. Inverse-square distance attenuation (solid angle scaling)
        effective_dist = np.maximum(dists, min_distance_mm)
        dist_weight = 1.0 / (effective_dist ** 2)  # (M,)

        # Total intensity per ommatidium: sum over points M
        frame_resp = np.sum(rf_weights * dist_weight[:, np.newaxis], axis=0)
        intensities[fr] = frame_resp.astype(np.float32)

    return intensities
