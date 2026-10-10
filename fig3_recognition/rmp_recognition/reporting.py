"""Result reporting: console summary, CSV export, robustness figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCHEME_COLORS: dict[str, str] = {'RMP': '#1f77b4', 'Uniform': '#2ca02c'}


def print_summary(results: dict[str, dict]) -> None:
    """Print the manuscript-level recognition conclusions."""
    line = '=' * 70
    print(f'\n{line}')
    print('  Recognition results (RMP vs uniform benchmark)')
    print(line)
    header = (f"{'Scheme':<10} {'Inputs':<8} {'Dir Acc':<10} "
              f"{'Letter Acc':<12} {'Let σ=0.2':<12} {'Let σ=0.4':<12}")
    print(header)
    print('-' * len(header))
    inputs = {'RMP': '17', 'Uniform': '81'}
    for scheme, r in results.items():
        rb = r['robust']
        i02 = rb['noise_std'].index(0.2)
        i04 = rb['noise_std'].index(0.4)
        print(f"{scheme:<10} {inputs[scheme]:<8} {r['acc_dir']:<10.4f} "
              f"{r['acc_let']:<12.4f} {rb['let_acc'][i02]:<12.4f} "
              f"{rb['let_acc'][i04]:<12.4f}")


def save_csv(results: dict[str, dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for scheme, r in results.items():
        rb = r['robust']
        for i, std in enumerate(rb['noise_std']):
            rows.append({
                'Scheme': scheme, 'Noise_Sigma': std,
                'Direction_Accuracy': rb['dir_acc'][i],
                'Direction_Std': rb['dir_acc_std'][i],
                'Letter_Accuracy': rb['let_acc'][i],
                'Letter_Std': rb['let_acc_std'][i],
            })
    path = output_dir / 'noise_robustness.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  Saved: {path}')

    rows = [{'Scheme': s, 'Direction_Accuracy': results[s]['acc_dir'],
             'Letter_Accuracy': results[s]['acc_let']}
            for s in results]
    path = output_dir / 'clean_accuracy.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  Saved: {path}')


def plot_robustness(results: dict[str, dict], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    kw = dict(marker='o', capsize=3, linewidth=2, markersize=6)
    for scheme, r in results.items():
        rb = r['robust']
        axes[0].errorbar(rb['noise_std'], rb['dir_acc'],
                         yerr=rb['dir_acc_std'],
                         color=SCHEME_COLORS.get(scheme), label=scheme, **kw)
        axes[1].errorbar(rb['noise_std'], rb['let_acc'],
                         yerr=rb['let_acc_std'],
                         color=SCHEME_COLORS.get(scheme), label=scheme, **kw)

    for ax, title, chance in [(axes[0], 'Direction', 0.5),
                              (axes[1], 'Letter', 0.25)]:
        ax.axhline(chance, color='grey', linestyle='--', alpha=0.5,
                   label='Chance')
        ax.set_title(f'{title} — Noise Robustness')
        ax.set_xlabel('Noise σ')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.5)
        ax.legend()

    fig.tight_layout()
    path = output_dir / 'noise_robustness.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {path}')
