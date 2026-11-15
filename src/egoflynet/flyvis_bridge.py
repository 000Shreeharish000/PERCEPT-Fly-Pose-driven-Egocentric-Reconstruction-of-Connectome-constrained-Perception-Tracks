"""Optional bridge to connectome-constrained visual models via flyvis.

References:
    Lappalainen, J. K., Tschopp, F., Grover, D., et al. (2024).
    Connectome-constrained networks predict neural activity across the
    Drosophila visual system. Nature, 634, 1132-1140.
    Code: https://github.com/turagalab/flyvis (MIT License)
"""

from __future__ import annotations

import importlib.util
from typing import Any, Sequence

import numpy as np


def is_flyvis_available() -> bool:
    """Check whether the flyvis package is installed in the current environment."""
    return importlib.util.find_spec("flyvis") is not None


def get_flyvis_info() -> dict[str, str]:
    """Return citation and metadata regarding the flyvis package."""
    return {
        "title": "Connectome-constrained networks predict neural activity across the Drosophila visual system",
        "authors": "Janne K. Lappalainen, Fabian D. Tschopp, Dhruv Grover, et al.",
        "journal": "Nature 634, 1132–1140 (2024)",
        "doi": "10.1038/s41586-024-07939-3",
        "license": "MIT",
        "repository": "https://github.com/turagalab/flyvis",
    }


def simulate_connectome_vision(
    intensities: np.ndarray,
    cell_types: Sequence[str] | None = None,
    dt_sec: float = 1.0 / 60.0,
    network_id: str = "flow/0000/000",
) -> dict[str, np.ndarray]:
    """Run egocentric ommatidial signals through a pretrained flyvis connectome network.

    NOTE: flyvis is an optional dependency. If not installed, calling this function
    will raise an informative ImportError.

    Args:
        intensities: (n_frames, n_ommatidia) array of visual input values.
        cell_types: List of cell types to query, e.g. ["T4a", "T4b", "T5a", "T5b"].
        dt_sec: Time step duration in seconds.
        network_id: Pretrained model identifier within flyvis.

    Returns:
        Dictionary mapping cell-type names to (n_frames, n_cells) activation arrays.
    """
    if not is_flyvis_available():
        raise ImportError(
            "The 'flyvis' library is required to simulate connectome-constrained neural responses.\n"
            "To install it, run:\n"
            "    pip install 'egoflynet[flyvis]'\n"
            "or refer to the flyvis repository at https://github.com/turagalab/flyvis.\n\n"
            "Reference: Lappalainen et al., Nature (2024). Connectome-constrained networks "
            "predict neural activity across the Drosophila visual system."
        )

    # Lazy imports
    import flyvis  # type: ignore
    import torch  # type: ignore

    targets = list(cell_types or ["T4a", "T4b", "T5a", "T5b"])

    # Load model via flyvis public API
    model = flyvis.load_model(network_id)
    model.eval()

    # Convert intensities to torch tensor (batch=1, time=n_frames, ommatidia=n_ommatidia)
    inp_tensor = torch.from_numpy(intensities).float().unsqueeze(0)

    with torch.no_grad():
        outputs: Any = model(inp_tensor)

    results: dict[str, np.ndarray] = {}
    for ct in targets:
        if hasattr(outputs, ct):
            act = getattr(outputs, ct).squeeze(0).cpu().numpy()
            results[ct] = act
        elif isinstance(outputs, dict) and ct in outputs:
            results[ct] = outputs[ct].squeeze(0).cpu().numpy()
        else:
            # Fallback placeholder if cell type not resolved
            results[ct] = np.zeros((intensities.shape[0], 1), dtype=np.float32)

    return results

