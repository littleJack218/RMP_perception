"""Evaluation protocol: frozen normalization, clean accuracy, robustness sweep.

The split and normalization mirror the protocol used when the checkpoints
were trained: a combination-level train/test split (25% held out,
stratified by letter x direction, seed 42) and per-dimension min/max
normalization taken from the clean training split.  This package ships the
held-out test split and the frozen normalization constants, so evaluation
requires no re-splitting; with the default seed the results are
deterministic and reproduce the manuscript values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from .data import add_pixel_noise, load_test_set
from .model import MultiTaskMLP
from .utils import set_seed

SCHEMES: tuple[str, ...] = ('RMP', 'Uniform')
HIDDEN_DIMS: tuple[int, ...] = (32, 32)

DEFAULT_SEED = 42
DEFAULT_NOISE_LEVELS: tuple[float, ...] = (
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
)
DEFAULT_REPEATS = 10


def evaluate_scheme(
    scheme: str,
    checkpoint: str | Path,
    data_dir: str | Path = 'data',
    device: str = 'cpu',
    seed: int = DEFAULT_SEED,
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    repeats: int = DEFAULT_REPEATS,
) -> dict:
    """Evaluate one scheme's checkpoint; returns accuracies and robustness.

    Returns a dict with keys ``acc_dir``, ``acc_let`` and ``robust`` (the
    latter holding per-sigma mean/std accuracies of both tasks).
    """
    set_seed(seed)

    X_te, yl_te, yd_te, _, X_min, X_max = load_test_set(scheme, data_dir)
    X_min = X_min.astype(np.float32)[None, :]
    X_max = X_max.astype(np.float32)[None, :]

    def _normalise(X: np.ndarray) -> np.ndarray:
        return (X - X_min) / (X_max - X_min + 1e-8)

    model = MultiTaskMLP(input_dim=X_te.shape[1],
                         hidden_dims=HIDDEN_DIMS).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device)['model_state_dict'])
    model.eval()

    def _predict(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xt = torch.tensor(_normalise(X), dtype=torch.float32).to(device)
        with torch.no_grad():
            od, ol = model(Xt)
        return od.argmax(1).cpu().numpy(), ol.argmax(1).cpu().numpy()

    # Clean test accuracy
    pred_dir, pred_let = _predict(X_te)
    acc_dir = float(accuracy_score(yd_te, pred_dir))
    acc_let = float(accuracy_score(yl_te, pred_let))

    # Noise robustness sweep
    robust = {
        'noise_std': list(noise_levels),
        'dir_acc': [], 'dir_acc_std': [],
        'let_acc': [], 'let_acc_std': [],
    }
    for std in noise_levels:
        dir_accs, let_accs = [], []
        for _ in range(repeats):
            X_noisy = add_pixel_noise(X_te, scheme, std)
            pd_, pl_ = _predict(X_noisy)
            dir_accs.append(accuracy_score(yd_te, pd_))
            let_accs.append(accuracy_score(yl_te, pl_))
        robust['dir_acc'].append(float(np.mean(dir_accs)))
        robust['dir_acc_std'].append(float(np.std(dir_accs)))
        robust['let_acc'].append(float(np.mean(let_accs)))
        robust['let_acc_std'].append(float(np.std(let_accs)))

    return {'acc_dir': acc_dir, 'acc_let': acc_let, 'robust': robust}


def evaluate_all(
    checkpoint_dir: str | Path = 'checkpoints',
    data_dir: str | Path = 'data',
    device: str = 'cpu',
    **kwargs,
) -> dict[str, dict]:
    """Evaluate every scheme against its checkpoint."""
    return {
        scheme: evaluate_scheme(
            scheme,
            checkpoint=Path(checkpoint_dir) / f'{scheme.lower()}_model.pt',
            data_dir=data_dir,
            device=device,
            **kwargs)
        for scheme in SCHEMES
    }
