"""I/O routines for loading and validating multi-fly pose tracking data."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


STANDARD_KEYPOINTS = ["head", "thorax", "abdomen_tip", "wing_left", "wing_right"]


@dataclass
class PoseData:
    """Multi-animal pose tracking data container.

    Attributes:
        tracks: Array of shape (n_frames, n_flies, n_keypoints, n_coords).
                n_coords should be 2 (x, y) or 3 (x, y, z).
        keypoint_names: Names corresponding to the keypoint dimension.
        fps: Sampling rate in frames per second.
        timestamps: Optional 1D array of time values in seconds for each frame.
    """

    tracks: np.ndarray
    keypoint_names: list[str] = field(default_factory=lambda: list(STANDARD_KEYPOINTS))
    fps: float = 60.0
    timestamps: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.tracks = np.asarray(self.tracks, dtype=np.float64)
        self.validate()
        if self.timestamps is None:
            self.timestamps = np.arange(self.n_frames, dtype=np.float64) / self.fps

    @property
    def n_frames(self) -> int:
        return int(self.tracks.shape[0])

    @property
    def n_flies(self) -> int:
        return int(self.tracks.shape[1])

    @property
    def n_keypoints(self) -> int:
        return int(self.tracks.shape[2])

    @property
    def n_coords(self) -> int:
        return int(self.tracks.shape[3])

    @property
    def is_3d(self) -> bool:
        return self.n_coords == 3

    def get_keypoint(self, fly_idx: int, keypoint_name: str) -> np.ndarray:
        """Extract trajectory of a specific keypoint for a given fly.

        Returns:
            Array of shape (n_frames, n_coords).
        """
        if fly_idx < 0 or fly_idx >= self.n_flies:
            raise IndexError(f"Fly index {fly_idx} out of range [0, {self.n_flies - 1}].")
        if keypoint_name not in self.keypoint_names:
            raise KeyError(
                f"Keypoint '{keypoint_name}' not found. Available keypoints: {self.keypoint_names}"
            )
        kp_idx = self.keypoint_names.index(keypoint_name)
        return self.tracks[:, fly_idx, kp_idx, :]

    def validate(self) -> None:
        """Validate shapes and invariants of the pose tracking data."""
        if self.tracks.ndim != 4:
            raise ValueError(
                f"Expected tracks array of rank 4 (frames, flies, keypoints, coords), "
                f"got rank {self.tracks.ndim} with shape {self.tracks.shape}."
            )
        if self.n_coords not in (2, 3):
            raise ValueError(
                f"Coordinate dimension must be 2 (x, y) or 3 (x, y, z), got {self.n_coords}."
            )
        if len(self.keypoint_names) != self.n_keypoints:
            raise ValueError(
                f"Mismatch: {len(self.keypoint_names)} keypoint names provided, "
                f"but tracks array has {self.n_keypoints} keypoints."
            )
        if self.n_frames == 0:
            raise ValueError("PoseData contains 0 frames.")
        if self.n_flies < 1:
            raise ValueError("PoseData must contain at least 1 fly.")
        if self.fps <= 0:
            raise ValueError(f"FPS must be positive, got {self.fps}.")

    def as_3d(self) -> PoseData:
        """Return a copy with coordinates promoted to 3D (z=0 if previously 2D)."""
        if self.is_3d:
            return self
        z_col = np.zeros((*self.tracks.shape[:3], 1), dtype=self.tracks.dtype)
        tracks_3d = np.concatenate([self.tracks, z_col], axis=-1)
        return PoseData(
            tracks=tracks_3d,
            keypoint_names=list(self.keypoint_names),
            fps=self.fps,
            timestamps=None if self.timestamps is None else self.timestamps.copy(),
        )


def load_csv(path: Path | str, keypoint_names: Sequence[str] | None = None, fps: float = 60.0) -> PoseData:
    """Load pose data from a CSV file.

    Supports:
      1. Long format: columns [frame, fly, keypoint, x, y[, z]]
      2. Wide format: columns [frame, fly0_head_x, fly0_head_y, ...]
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV file {file_path} is empty.")
        rows = [r for r in reader if r]

    if not rows:
        raise ValueError(f"CSV file {file_path} contains no data rows.")

    header_lower = [h.strip().lower() for h in header]

    # Detect Long Format: contains 'frame', 'fly' (or 'fly_id'), 'keypoint', 'x', 'y'
    if {"frame", "keypoint", "x", "y"}.issubset(set(header_lower)):
        frame_idx = header_lower.index("frame")
        fly_idx = header_lower.index("fly") if "fly" in header_lower else header_lower.index("fly_id") if "fly_id" in header_lower else None
        kp_idx = header_lower.index("keypoint")
        x_idx = header_lower.index("x")
        y_idx = header_lower.index("y")
        has_z = "z" in header_lower
        z_idx = header_lower.index("z") if has_z else None

        parsed_rows = []
        unique_frames: set[int] = set()
        unique_flies: set[int] = set()
        unique_kps: list[str] = []

        for row in rows:
            fr = int(float(row[frame_idx]))
            fl = int(float(row[fly_idx])) if fly_idx is not None else 0
            kp = row[kp_idx].strip()
            x = float(row[x_idx])
            y = float(row[y_idx])
            z = float(row[z_idx]) if z_idx is not None else 0.0

            unique_frames.add(fr)
            unique_flies.add(fl)
            if kp not in unique_kps:
                unique_kps.append(kp)
            parsed_rows.append((fr, fl, kp, x, y, z))

        n_frames = max(unique_frames) + 1
        n_flies = max(unique_flies) + 1
        n_kps = len(unique_kps)
        n_coords = 3 if has_z else 2

        tracks = np.full((n_frames, n_flies, n_kps, n_coords), np.nan, dtype=np.float64)
        for fr, fl, kp, x, y, z in parsed_rows:
            ki = unique_kps.index(kp)
            if has_z:
                tracks[fr, fl, ki] = [x, y, z]
            else:
                tracks[fr, fl, ki] = [x, y]

        return PoseData(tracks=tracks, keypoint_names=unique_kps, fps=fps)

    # Detect Wide Format: header has columns like fly0_head_x, fly0_head_y or head_x, head_y
    raw_data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if raw_data.ndim == 1:
        raw_data = raw_data[np.newaxis, :]

    # Look for structured column names
    kps: list[str] = []
    fly_indices: set[int] = set()
    coords_found: dict[tuple[int, str], dict[str, int]] = {}

    for c_idx, col in enumerate(header_lower):
        parts = col.split("_")
        if len(parts) >= 3 and parts[0].startswith("fly"):
            try:
                fl = int(parts[0].replace("fly", ""))
                coord = parts[-1]
                kp = "_".join(parts[1:-1])
                if coord in ("x", "y", "z"):
                    fly_indices.add(fl)
                    if kp not in kps:
                        kps.append(kp)
                    coords_found.setdefault((fl, kp), {})[coord] = c_idx
            except ValueError:
                continue

    if coords_found:
        n_frames = len(raw_data)
        n_flies = max(fly_indices) + 1
        n_kps = len(kps)
        has_z = any("z" in v for v in coords_found.values())
        n_coords = 3 if has_z else 2

        tracks = np.full((n_frames, n_flies, n_kps, n_coords), np.nan, dtype=np.float64)
        for (fl, kp), cdict in coords_found.items():
            ki = kps.index(kp)
            tracks[:, fl, ki, 0] = raw_data[:, cdict["x"]]
            tracks[:, fl, ki, 1] = raw_data[:, cdict["y"]]
            if has_z and "z" in cdict:
                tracks[:, fl, ki, 2] = raw_data[:, cdict["z"]]

        return PoseData(tracks=tracks, keypoint_names=kps, fps=fps)

    # Simple fallback: assume standard keypoints for 2 flies, 2D coordinates
    kps = list(keypoint_names or STANDARD_KEYPOINTS)
    n_kps = len(kps)
    # Check if raw_data columns can be reshaped to (n_frames, n_flies, n_kps, 2)
    total_cols = raw_data.shape[1]
    # If first column is frame number, skip it
    if total_cols % (n_kps * 2) == 1:
        raw_data = raw_data[:, 1:]
        total_cols -= 1

    n_flies = total_cols // (n_kps * 2)
    if n_flies >= 1 and total_cols == n_flies * n_kps * 2:
        tracks = raw_data.reshape(len(raw_data), n_flies, n_kps, 2)
        return PoseData(tracks=tracks, keypoint_names=kps, fps=fps)

    raise ValueError(
        f"Unable to automatically infer pose format from CSV columns: {header}. "
        "Expected long format (frame, fly, keypoint, x, y) or wide format (fly0_head_x, ...)."
    )


def save_csv(data: PoseData, path: Path | str) -> None:
    """Save PoseData to a CSV file in standard wide format."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["frame"]
    for fl in range(data.n_flies):
        for kp in data.keypoint_names:
            header.append(f"fly{fl}_{kp}_x")
            header.append(f"fly{fl}_{kp}_y")
            if data.is_3d:
                header.append(f"fly{fl}_{kp}_z")

    rows: list[list[str]] = [header]
    for fr in range(data.n_frames):
        row: list[str] = [str(fr)]
        for fl in range(data.n_flies):
            for ki in range(data.n_keypoints):
                coords = data.tracks[fr, fl, ki]
                row.extend([f"{c:.4f}" for c in coords])
        rows.append(row)

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def load_sleap_h5(path: Path | str, fps: float = 60.0) -> PoseData:
    """Load pose tracking data from a SLEAP HDF5 analysis export.

    Args:
        path: Path to .h5 or .analysis.h5 file.
        fps: Sampling frame rate.

    Returns:
        PoseData container.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"SLEAP h5 file not found: {file_path}")

    try:
        import h5py
    except ImportError as err:
        raise ImportError(
            "h5py is required to load SLEAP .h5 files. "
            "Install it via `pip install h5py` or `pip install 'egoflynet[dev]'`."
        ) from err

    with h5py.File(file_path, "r") as f:
        # Standard SLEAP h5 structure:
        # 'tracks': shape (n_nodes, n_coords, n_tracks, n_frames) or (n_frames, n_tracks, n_nodes, n_coords)
        if "tracks" not in f:
            raise KeyError(f"Expected dataset 'tracks' in {file_path}. Found keys: {list(f.keys())}")

        raw_tracks = np.array(f["tracks"])
        # Node names
        if "node_names" in f:
            node_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in f["node_names"]]
        else:
            node_names = [f"node_{i}" for i in range(raw_tracks.shape[0])]

        # Transpose to (n_frames, n_flies, n_keypoints, n_coords)
        if raw_tracks.ndim == 4:
            # SLEAP typically outputs (n_nodes, n_coords, n_tracks, n_frames)
            if raw_tracks.shape[1] in (2, 3) and raw_tracks.shape[0] == len(node_names):
                # (nodes, coords, tracks, frames) -> (frames, tracks, nodes, coords)
                tracks = np.transpose(raw_tracks, (3, 2, 0, 1))
            else:
                # Already in canonical or alternative ordering
                tracks = raw_tracks
        else:
            raise ValueError(f"Unexpected SLEAP tracks rank: {raw_tracks.ndim}")

    return PoseData(tracks=tracks, keypoint_names=node_names, fps=fps)

