"""EgoFlyNet End-to-End Demo Script.

Demonstrates:
1. Synthetic 2-fly courtship trajectory generation.
2. Focal-fly body coordinate frame computation.
3. Hexagonal ommatidia compound eye lattice modeling.
4. Egocentric projection and ommatidial excitation.
5. Visualizing top-down, egocentric, and temporal traces.
"""

from pathlib import Path
import numpy as np

from egoflynet import (
    generate_courtship_trajectory,
    compute_body_frame,
    create_hexagonal_eye_lattice,
    project_partner_onto_eyes,
    is_flyvis_available,
    simulate_connectome_vision,
)
from egoflynet.viz import plot_frame, render_animation


def main():
    out_dir = Path(__file__).resolve().parent / "demo_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Generating synthetic courtship trajectory (120 frames @ 60 Hz)...")
    pose = generate_courtship_trajectory(n_frames=120, fps=60.0, seed=42)
    print(f"       Generated {pose.n_frames} frames for {pose.n_flies} flies.")

    print("Step 2: Computing focal fly body frame...")
    body_frame = compute_body_frame(pose, focal_fly_idx=0)
    print(f"       Head origin: {body_frame.origin[0]}")
    print(f"       Forward vector: {body_frame.forward[0]}")

    print("Step 3: Constructing hexagonal ommatidia lattice (~720 ommatidia)...")
    lattice = create_hexagonal_eye_lattice(n_cols=24, n_rows=15)
    print(f"       Total ommatidia: {lattice.n_ommatidia} (Left: {len(lattice.left_indices)}, Right: {len(lattice.right_indices)})")

    print("Step 4: Projecting partner fly into egocentric visual field...")
    intensities = project_partner_onto_eyes(pose, body_frame, lattice, dense_body=True)
    np.save(out_dir / "ommatidia_intensities.npy", intensities)
    print(f"       Excitation matrix shape: {intensities.shape}")
    print(f"       Peak excitation: {np.max(intensities):.3f}")

    print("Step 5: Optional connectome model check (flyvis)...")
    if is_flyvis_available():
        print("       flyvis detected. Running connectome simulation...")
        responses = simulate_connectome_vision(intensities)
        for cell_type, resp in responses.items():
            print(f"       - {cell_type}: max activation {np.max(resp):.3f}")
    else:
        print("       flyvis not installed. Skipping connectome simulation (optional extra).")

    print("Step 6: Saving visualization figures...")
    preview_file = out_dir / "demo_preview.png"
    plot_frame(pose, body_frame, lattice, intensities, frame_idx=45, save_path=preview_file)
    print(f"       Saved preview: {preview_file}")

    gif_file = out_dir / "demo_animation.gif"
    print(f"       Rendering animation ({gif_file})...")
    render_animation(pose, body_frame, lattice, intensities, output_path=gif_file, max_frames=50)
    print(f"       Saved animation: {gif_file}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
