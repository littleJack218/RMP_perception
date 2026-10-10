# Checkpoints

| File | Network | Role |
|---|---|---|
| `mixed_cnn.pth` | SimpleCNN (4 ch, 0.45 M params) | mixed-scale readout; test accuracy 97.3% (109/112) |
| `single_10Hz_+0V.pth` | MLP (1.80 M params) | uniform single-scale baseline (10 Hz / 0 V), 86.6% (97/112) |
| `mixed_cnn_noise_robust.pth` | SimpleCNN (4 ch) | optional noise-robust variant, same clean accuracy (Fig. S25) |

Reproduced by `python run_inference.py` from the package root.
