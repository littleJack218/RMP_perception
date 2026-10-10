#!/usr/bin/env python
"""Evaluate the pre-trained RMP / uniform recognition models.

Examples
--------
    python evaluate.py                       # defaults, CPU/GPU auto
    python evaluate.py --device cpu
    python evaluate.py --output-dir outputs/demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rmp_recognition import evaluate_all
from rmp_recognition.reporting import plot_robustness, print_summary, save_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate pre-trained RMP / uniform recognition models '
                    '(clean accuracy + Gaussian-noise robustness).')
    p.add_argument('--data-dir', type=Path, default=Path('data'),
                   help='directory containing the held-out test-set '
                        'archives (default: data)')
    p.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'),
                   help='directory containing {rmp,uniform}_model.pt '
                        '(default: checkpoints)')
    p.add_argument('--output-dir', type=Path, default=Path('outputs'),
                   help='directory for CSV/figure outputs (default: outputs)')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available()
                   else 'cpu', help='torch device (default: auto)')
    p.add_argument('--seed', type=int, default=42,
                   help='evaluation seed (default: 42, the manuscript value)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f'Device: {args.device}')
    print(f'Data:   {args.data_dir}/    Checkpoints: {args.checkpoint_dir}/')

    results = evaluate_all(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed,
    )

    print_summary(results)
    print(f'\nWriting outputs to {args.output_dir}/')
    save_csv(results, args.output_dir)
    plot_robustness(results, args.output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
