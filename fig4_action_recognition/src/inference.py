"""Load the packaged test split and checkpoints, and print the paper's
Figure-4 headline metrics.  Test-set-only package: nothing to fit and no
split to reproduce — the held-out split (112 of 372 encoded samples, 30%
held out with random state 42) is frozen in test_set/ in split order;
sample_id i corresponds to row `sample = i` of the Source Data sheet
`mechanism_persample`.
"""
import io
import json
import os

import numpy as np
import torch

from .models import MLP, SimpleCNN, n_params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, 'test_set')
WEIGHTS = os.path.join(ROOT, 'weights')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABELS = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk',
          'wave1', 'wave2']


def load_state(path):
    """Read a checkpoint via an in-memory buffer (torch's C++ zip reader
    fails on non-ASCII Windows paths)."""
    with open(path, 'rb') as fh:
        return torch.load(io.BytesIO(fh.read()), map_location=DEVICE)


def predict(model, X, batch=8):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32).to(DEVICE)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    yte = np.load(os.path.join(TEST, 'y_test.npy'))
    X4te = np.load(os.path.join(TEST, 'X_test_mixed_4ch.npy'))
    Xste = np.load(os.path.join(TEST, 'X_test_single_10Hz_+0V.npy'))

    mixed = SimpleCNN(4, (100, 60)).to(DEVICE)
    mixed.load_state_dict(load_state(os.path.join(WEIGHTS, 'mixed_cnn.pth')))
    single = MLP(6000).to(DEVICE)
    single.load_state_dict(load_state(os.path.join(WEIGHTS,
                                                   'single_10Hz_+0V.pth')))

    predR = predict(mixed, X4te)
    predS = predict(single, Xste)
    accR = (predR == yte).mean() * 100
    accS = (predS == yte).mean() * 100
    pm, ps = n_params(mixed), n_params(single)

    percls = {k: np.array([(p[yte == i] == i).mean() * 100
                           for i in range(10)])
              for k, p in [('RMP', predR), ('single', predS)]}

    W = 68
    print('=' * W)
    print(' RMP mixed-timescale action recognition - paper metrics')
    print('=' * W)
    print(f' mixed-scale (RMP, 4-timescale CNN) : {accR:6.2f}%   '
          f'[paper 97.3%]')
    print(f' uniform single-scale baseline      : {accS:6.2f}%   '
          f'[paper 86.6%]')
    print(f' improvement                        : '
          f'+{accR - accS:.1f} points [paper +10.7]')
    print(f' parameters                         : {pm/1e6:.2f} M vs '
          f'{ps/1e6:.2f} M  ({ps/pm:.2f}x fewer) [paper 3.99x]')
    print('=' * W)

    print('\n per-class accuracy (%)')
    print('encoding      ' + ' '.join(f'{l:>6}' for l in LABELS))
    print(f'{"single 10Hz":>13}  ' + ' '.join(f'{v:6.1f}' for v in percls['single']))
    print(f'{"RMP fused":>13}  ' + ' '.join(f'{v:6.1f}' for v in percls['RMP']))

    out = {'mixed_acc': accR, 'single_baseline_acc': accS,
           'gain_points': accR - accS,
           'params_mixed': pm, 'params_single': ps,
           'per_class': {k: [float(v) for v in percls[k]]
                         for k in percls}}
    with open(os.path.join(ROOT, 'metrics_summary.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('\nmetrics_summary.json written.')
