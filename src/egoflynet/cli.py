"""Command-line interface for the EgoFlyNet pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .io import load_csv, load_sleap_h5, save_csv
from .synthetic import generate_courtship_trajectory
from .body import compute_body_frame
from .eye import create_hexagonal_eye_lattice
from .projection import project_partner_onto_eyes
from .flyvis_bridge import is_flyvis_available, simulate_connectome_vision
from .viz import plot_frame, render_animation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="egoflynet",
        description="EgoFlyNet: Reconstruct Drosophila egocentric visual input from multi-fly pose tracking.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Run egocentric visual reconstruction pipeline.")
    run_p.add_argument(
        "--input",
        "-i",
        type=str,
        default="synthetic",
        help="Input pose data path (.csv, .h5) or 'synthetic' (default).",
    )
    run_p.add_argument(
        "--out",
        "-o",
        type=str,
        default="outputs",
        help="Output directory for generated artifacts.",
    )
    run_p.add_argument("--frames", type=int, default=120, help="Number of frames for synthetic data.")
    run_p.add_argument("--fps", type=float, default=60.0, help="Frame rate in Hz.")
    run_p.add_argument("--seed", type=int, default=42, help="Random seed for synthetic generation.")
    run_p.add_argument("--gif", action="store_true", default=True, help="Render animated GIF visualization.")
    run_p.add_argument("--no-gif", dest="gif", action="store_false", help="Skip animated GIF visualization.")
    run_p.add_argument("--flyvis", action="store_true", help="Run optional flyvis neural model if installed.")

    return parser


def run_pipeline(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EgoFlyNet: Pose-Driven Egocentric Visual Reconstruction")
    print("=" * 60)

    # 1. Load or Generate Pose Data
    if args.input.lower() == "synthetic":
        print(f"[*] Generating {args.frames} frames of synthetic courtship trajectory (seed={args.seed})...")
        pose = generate_courtship_trajectory(n_frames=args.frames, fps=args.fps, seed=args.seed)
        save_csv(pose, out_dir / "synthetic_pose.csv")
    else:
        in_path = Path(args.input)
        print(f"[*] Loading pose data from {in_path}...")
        if in_path.suffix in (".h5", ".hdf5"):
            pose = load_sleap_h5(in_path, fps=args.fps)
        else:
            pose = load_csv(in_path, fps=args.fps)

    print(f"    - Frames: {pose.n_frames}")
    print(f"    - Flies: {pose.n_flies}")
    print(f"    - Keypoints: {pose.keypoint_names}")

    # 2. Reconstruct Focal Fly Body Frame
    print("[*] Computing focal-fly orthonormal body coordinate frame...")
    body_frame = compute_body_frame(pose, focal_fly_idx=0)
    print(f"    - Heading span: yaw [{np.min(body_frame.yaw):.2f}, {np.max(body_frame.yaw):.2f}] rad")

    # 3. Create Hexagonal Ommatidia Lattice
    print("[*] Creating hexagonal compound eye lattice (~720 ommatidia)...")
    lattice = create_hexagonal_eye_lattice()
    print(f"    - Left eye ommatidia: {len(lattice.left_indices)}")
    print(f"    - Right eye ommatidia: {len(lattice.right_indices)}")

    # 4. Egocentric Projection and Ommatidial Rasterization
    print("[*] Projecting partner fly into egocentric visual field...")
    intensities = project_partner_onto_eyes(pose, body_frame, lattice, focal_idx=0, partner_idx=1)
    np.save(out_dir / "ommatidia_intensities.npy", intensities)
    print(f"    - Output excitation array shape: {intensities.shape}")
    print(f"    - Peak visual intensity: {np.max(intensities):.4f}")

    # 5. Optional Flyvis Connectome Simulation
    if args.flyvis:
        print("[*] Running optional connectome-constrained flyvis model...")
        if is_flyvis_available():
            resp = simulate_connectome_vision(intensities)
            for ct, val in resp.items():
                np.save(out_dir / f"response_{ct}.npy", val)
                print(f"    - {ct} activity recorded (shape {val.shape})")
        else:
            print("    [!] flyvis is not installed in the current environment.")
            print("        Install with `pip install 'egoflynet[flyvis]'` to enable this feature.")

    # 6. Visualizations
    mid_frame = min(40, pose.n_frames - 1)
    preview_path = out_dir / "reconstruction_preview.png"
    print(f"[*] Saving static preview figure to {preview_path}...")
    plot_frame(pose, body_frame, lattice, intensities, frame_idx=mid_frame, save_path=preview_path)

    if args.gif:
        gif_path = out_dir / "reconstruction.gif"
        print(f"[*] Rendering animated GIF to {gif_path} (this may take a few seconds)...")
        render_animation(pose, body_frame, lattice, intensities, output_path=gif_path, max_frames=60)
        print(f"    - Saved animation: {gif_path}")

    # 7. Write Summary JSON
    summary = {
        "n_frames": pose.n_frames,
        "n_flies": pose.n_flies,
        "keypoints": pose.keypoint_names,
        "n_ommatidia": lattice.n_ommatidia,
        "peak_intensity": float(np.max(intensities)),
        "mean_intensity": float(np.mean(intensities)),
        "fps": pose.fps,
        "artifacts": [
            "ommatidia_intensities.npy",
            "reconstruction_preview.png",
            "reconstruction.gif" if args.gif else None,
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[✓] Pipeline completed successfully!")
    print(f"    Artifacts saved to: {out_dir.resolve()}\n")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
