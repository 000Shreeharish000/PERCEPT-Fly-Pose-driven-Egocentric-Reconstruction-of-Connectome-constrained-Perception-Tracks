"""EgoFlyNet: Pose-driven egocentric visual reconstruction for Drosophila."""

from .io import PoseData, load_csv, load_sleap_h5
from .synthetic import generate_courtship_trajectory
from .body import FlyBodyFrame, compute_body_frame
from .eye import OmmatidiaLattice, create_hexagonal_eye_lattice
from .projection import project_partner_onto_eyes
from .flyvis_bridge import is_flyvis_available, simulate_connectome_vision

__version__ = "0.1.0"

__all__ = [
    "PoseData",
    "load_csv",
    "load_sleap_h5",
    "generate_courtship_trajectory",
    "FlyBodyFrame",
    "compute_body_frame",
    "OmmatidiaLattice",
    "create_hexagonal_eye_lattice",
    "project_partner_onto_eyes",
    "is_flyvis_available",
    "simulate_connectome_vision",
]

