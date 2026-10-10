"""RMP mixed-timescale recognition: held-out test set and sensor noise model.

This release ships the frozen held-out test split (128 samples per scheme)
together with the per-dimension normalization constants that were computed
on the clean training split when the checkpoints were trained.  The full
512-sample training datasets are not part of the package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_FILES: dict[str, str] = {
    'RMP': 'rmp_test_set.npz',
    'Uniform': 'uniform_test_set.npz',
}

_SURROUND_REGIONS: list[tuple[int, int]] = [
    (0, 0), (0, 3), (0, 6),
    (3, 6), (6, 6), (6, 3), (6, 0), (3, 0),
]


# ---- Pixel <-> feature conversion ----

def rmp_17d_to_grid(feat: np.ndarray) -> np.ndarray:
    """Reconstruct a 9x9 grid from 17-dim RMP features."""
    grid = np.zeros((9, 9), dtype=np.float32)
    grid[3:6, 3:6] = feat[:9].reshape(3, 3)
    for k, (r, c) in enumerate(_SURROUND_REGIONS):
        grid[r:r + 3, c:c + 3] = feat[9 + k]
    return grid


def grid_to_rmp_17d(grid: np.ndarray) -> np.ndarray:
    """Extract 17-dim RMP features from a 9x9 grid (surround = 3x3 block avg)."""
    centre = grid[3:6, 3:6].ravel()
    surround = np.array([grid[r:r + 3, c:c + 3].mean()
                         for r, c in _SURROUND_REGIONS], dtype=np.float32)
    return np.concatenate([centre, surround])


def uniform_81d_to_grid(feat: np.ndarray) -> np.ndarray:
    return feat.reshape(9, 9).astype(np.float32)


def grid_to_uniform_81d(grid: np.ndarray) -> np.ndarray:
    return grid.ravel().astype(np.float32)


# ---- Sensor noise model ----

def add_pixel_noise(feat: np.ndarray, scheme: str, std: float) -> np.ndarray:
    """Add Gaussian noise at the 9x9 pixel level, then re-extract features.

    For the RMP, each peripheral super-pixel covers a 3x3 block and averages
    the pixel-level noise, reflecting the physical advantage of larger
    photodetectors.
    """
    if std == 0.0:
        return feat

    to_grid = rmp_17d_to_grid if scheme == 'RMP' else uniform_81d_to_grid
    to_feat = grid_to_rmp_17d if scheme == 'RMP' else grid_to_uniform_81d

    if feat.ndim == 1:
        grid = to_grid(feat)
        grid += np.random.normal(0, std, grid.shape).astype(np.float32)
        np.clip(grid, 0, None, out=grid)
        return to_feat(grid)

    out = np.zeros_like(feat)
    for i in range(len(feat)):
        grid = to_grid(feat[i])
        grid += np.random.normal(0, std, grid.shape).astype(np.float32)
        np.clip(grid, 0, None, out=grid)
        out[i] = to_feat(grid)
    return out


# ---- Held-out test set ----

def load_test_set(
    scheme: str,
    data_dir: str | Path = 'data',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Load the frozen held-out test split of one scheme.

    Returns ``(X, y_letter, y_direction, combo_info, norm_min, norm_max)``.
    ``combo_info`` columns are (letter, start_phase, direction, frame); the
    128 samples are the 16 held-out letter/start/direction combinations
    (8 frames each, combination-level split, stratified by letter x
    direction, seed 42) in dataset order.  ``norm_min`` / ``norm_max`` are
    the per-dimension min/max of the clean **training** split, frozen when
    the checkpoints were trained and shipped here so that evaluation
    reproduces the manuscript values exactly.
    """
    if scheme not in DATA_FILES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of "
                         f"{sorted(DATA_FILES)}")
    path = Path(data_dir) / DATA_FILES[scheme]
    with np.load(path) as d:
        return (d['X'], d['y_letter'], d['y_direction'], d['combo_info'],
                d['norm_min'], d['norm_max'])
