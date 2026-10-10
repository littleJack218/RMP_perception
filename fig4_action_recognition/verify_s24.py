#!/usr/bin/env python
"""Derive the released-run confusion matrices of Supplementary Fig. S24.

Runs the packaged checkpoints on the packaged test split (same protocol as
run_inference.py) and writes:
    s24_predictions.csv          per-sample predictions (112 x [true, single, RMP])
    s24_confusion_matrices.csv   the confusion matrices of Fig. S24 (mixed / single)
"""
import csv, io, os
import numpy as np, torch

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, ROOT)
from src.models import MLP, SimpleCNN

LABELS = ['bend', 'jack', 'jump', 'pjump', 'run', 'side', 'skip', 'walk',
          'wave1', 'wave2']
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_state(path):
    with open(path, 'rb') as fh:
        return torch.load(io.BytesIO(fh.read()), map_location=DEV)


def predict(model, X, bs=8):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs], dtype=torch.float32).to(DEV)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    T = os.path.join(ROOT, 'test_set')
    W = os.path.join(ROOT, 'weights')
    yte = np.load(os.path.join(T, 'y_test.npy'))
    mix = SimpleCNN(4, (100, 60)).to(DEV)
    mix.load_state_dict(load_state(os.path.join(W, 'mixed_cnn.pth')))
    predR = predict(mix, np.load(os.path.join(T, 'X_test_mixed_4ch.npy')))
    sgl = MLP(6000).to(DEV)
    sgl.load_state_dict(load_state(os.path.join(W, 'single_10Hz_+0V.pth')))
    predS = predict(sgl, np.load(os.path.join(T, 'X_test_single_10Hz_+0V.npy')))

    with open('s24_predictions.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['sample', 'true_label', 'action',
                    'pred_single_10Hz', 'pred_RMP'])
        for i in range(len(yte)):
            w.writerow([i, int(yte[i]), LABELS[int(yte[i])],
                        int(predS[i]), int(predR[i])])

    def cm(pred):
        M = np.zeros((10, 10), int)
        for t, p in zip(yte, pred):
            M[int(t), int(p)] += 1
        return M

    with open('s24_confusion_matrices.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['matrix', 'true_label', 'predicted_label', 'count'])
        for name, pred in [('mixed_RMP', predR), ('single_10Hz_+0V', predS)]:
            M = cm(pred)
            for i in range(10):
                for j in range(10):
                    if M[i, j]:
                        w.writerow([name, i, j, M[i, j]])
            print(f'{name}: trace {M.trace()}/112 = {M.trace()/112*100:.2f}%')

    print('written: s24_predictions.csv, s24_confusion_matrices.csv')


if __name__ == '__main__':
    main()
