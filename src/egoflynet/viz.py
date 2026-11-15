"""Visualization utilities: top-down arena, egocentric ommatidia grid, and response traces."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .io import PoseData
from .body import FlyBodyFrame
from .eye import OmmatidiaLattice


def plot_frame(
    pose: PoseData,
    body_frame: FlyBodyFrame,
    lattice: OmmatidiaLattice,
    intensities: np.ndarray,
    frame_idx: int,
    save_path: Path | str | None = None,
    focal_idx: int = 0,
    partner_idx: int = 1,
) -> plt.Figure:
    """Generate a 3-panel figure for a single frame.

    Panels:
      1. Top-down arena view showing focal fly orientation and partner trajectory.
      2. Egocentric ommatidial excitation grid (Azimuth vs Elevation) for both eyes.
      3. Total visual flux / contrast time series with marker at current frame.

    Args:
        pose: Pose data.
        body_frame: Reconstructed focal fly body frame.
        lattice: Ommatidia lattice.
        intensities: (n_frames, n_ommatidia) array of visual excitations.
        frame_idx: Index of the current frame to visualize.
        save_path: Optional path to save image file (e.g. .png).
        focal_idx: Focal fly index.
        partner_idx: Partner fly index.

    Returns:
        matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=100)
    fig.patch.set_facecolor("#12141a")

    # Styling colors
    text_color = "#e0e3eb"
    focal_color = "#4cc9f0"
    partner_color = "#f72585"
    grid_bg = "#1a1d26"

    for ax in axes:
        ax.set_facecolor(grid_bg)
        ax.tick_params(colors=text_color, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#333846")
        ax.title.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

    # ---------------- Panel 1: Top-Down Arena View ----------------
    ax1 = axes[0]
    ax1.set_title(f"Top-Down View (Frame {frame_idx})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (mm)", fontsize=10)
    ax1.set_ylabel("Y (mm)", fontsize=10)

    # Historical trails (last 30 frames)
    trail_start = max(0, frame_idx - 30)
    focal_trail = pose.tracks[trail_start : frame_idx + 1, focal_idx, 1, :2]  # thorax
    partner_trail = pose.tracks[trail_start : frame_idx + 1, partner_idx, 1, :2]

    if len(focal_trail) > 1:
        ax1.plot(focal_trail[:, 0], focal_trail[:, 1], color=focal_color, alpha=0.4, lw=1.5, ls="--")
    if len(partner_trail) > 1:
        ax1.plot(partner_trail[:, 0], partner_trail[:, 1], color=partner_color, alpha=0.4, lw=1.5, ls="--")

    # Current keypoints
    focal_kps = pose.tracks[frame_idx, focal_idx, :, :2]
    partner_kps = pose.tracks[frame_idx, partner_idx, :, :2]

    # Draw body segments for focal fly
    if pose.n_keypoints >= 3:
        ax1.plot(focal_kps[:3, 0], focal_kps[:3, 1], color=focal_color, lw=2.5, label="Focal fly")
        ax1.plot(partner_kps[:3, 0], partner_kps[:3, 1], color=partner_color, lw=2.5, label="Partner fly")

    ax1.scatter(focal_kps[:, 0], focal_kps[:, 1], color=focal_color, s=25, zorder=5)
    ax1.scatter(partner_kps[:, 0], partner_kps[:, 1], color=partner_color, s=25, zorder=5)

    # Focal heading vector
    focal_pos = body_frame.origin[frame_idx, :2]
    fwd_vec = body_frame.forward[frame_idx, :2] * 1.5
    ax1.annotate(
        "",
        xy=(focal_pos[0] + fwd_vec[0], focal_pos[1] + fwd_vec[1]),
        xytext=(focal_pos[0], focal_pos[1]),
        arrowprops=dict(arrowstyle="->", color="#fca311", lw=2.0),
    )

    ax1.legend(loc="upper right", facecolor="#1a1d26", edgecolor="#333846", labelcolor=text_color, fontsize=8)
    ax1.set_aspect("equal", adjustable="datalim")

    # ---------------- Panel 2: Egocentric Ommatidia Lattice ----------------
    ax2 = axes[1]
    ax2.set_title("Egocentric Compound Eyes (Azimuth / Elevation)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Azimuth (deg)", fontsize=10)
    ax2.set_ylabel("Elevation (deg)", fontsize=10)

    az_deg = np.rad2deg(lattice.azimuth)
    el_deg = np.rad2deg(lattice.elevation)
    cur_intensities = intensities[frame_idx]

    # Normalize intensity for visualization
    max_val = np.max(intensities) if np.max(intensities) > 0 else 1.0
    norm_intensities = cur_intensities / max_val

    # Background lattice
    ax2.scatter(az_deg, el_deg, c=norm_intensities, cmap="inferno", s=22, edgecolors="none", alpha=0.9)
    ax2.axvline(0, color="#ffffff", alpha=0.3, ls=":", lw=1, label="Frontal center")

    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-70, 70)
    ax2.text(-150, -60, "Left Eye", color=focal_color, fontsize=8, fontweight="bold")
    ax2.text(90, -60, "Right Eye", color=partner_color, fontsize=8, fontweight="bold")

    # ---------------- Panel 3: Temporal Flux Trace ----------------
    ax3 = axes[2]
    ax3.set_title("Total Visual Flux over Time", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.set_ylabel("Summed Excitation", fontsize=10)

    total_flux = np.sum(intensities, axis=-1)
    time_s = pose.timestamps if pose.timestamps is not None else np.arange(pose.n_frames) / pose.fps

    ax3.plot(time_s, total_flux, color="#06d6a0", lw=1.8)
    cur_time = time_s[frame_idx]
    ax3.axvline(cur_time, color="#ffd166", lw=1.5, ls="--", label="Current frame")
    ax3.scatter([cur_time], [total_flux[frame_idx]], color="#ffd166", s=40, zorder=5)

    ax3.legend(loc="upper right", facecolor="#1a1d26", edgecolor="#333846", labelcolor=text_color, fontsize=8)

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, facecolor=fig.get_facecolor(), edgecolor="none")

    return fig


def render_animation(
    pose: PoseData,
    body_frame: FlyBodyFrame,
    lattice: OmmatidiaLattice,
    intensities: np.ndarray,
    output_path: Path | str,
    fps: int = 20,
    max_frames: int | None = None,
) -> None:
    """Render a multi-frame animation (GIF) of the egocentric visual reconstruction.

    Args:
        pose: Pose tracking data.
        body_frame: Focal fly body coordinate frame.
        lattice: Ommatidia geometry.
        intensities: (n_frames, n_ommatidia) visual excitations.
        output_path: Output file path (.gif).
        fps: Animation playback speed.
        max_frames: Optional upper limit on rendered frames.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_frames = pose.n_frames if max_frames is None else min(pose.n_frames, max_frames)
    images: list[Image.Image] = []

    for fr in range(total_frames):
        fig = plot_frame(pose, body_frame, lattice, intensities, frame_idx=fr)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        images.append(img)

    if images:
        duration_ms = int(1000 / fps)
        images[0].save(
            out_file,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
