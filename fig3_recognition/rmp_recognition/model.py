"""Multi-task recognition network: shared MLP backbone with two heads."""

from __future__ import annotations

import torch
import torch.nn as nn

N_LETTER_CLASSES = 4      # N, U, D, T
N_DIRECTION_CLASSES = 2   # CW, CCW


class MultiTaskMLP(nn.Module):
    """Shared-backbone MLP with separate letter and direction heads.

    Parameters
    ----------
    input_dim:
        17 for the RMP (9 central fine pixels + 8 peripheral coarse pixels)
        or 81 for the equivalent-area uniform benchmark (9 x 9).
    hidden_dims:
        Widths of the shared ReLU backbone; must match the checkpoints.
    """

    def __init__(self, input_dim: int = 81,
                 hidden_dims: tuple[int, ...] = (32, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.direction_head = nn.Linear(hidden_dims[-1], N_DIRECTION_CLASSES)
        self.letter_head = nn.Linear(hidden_dims[-1], N_LETTER_CLASSES)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.backbone(x)
        return self.direction_head(f), self.letter_head(f)
