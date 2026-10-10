# rmp-recognition (Fig. 3)

Joint letter (4-class) and rotation-direction (2-class) recognition comparing
the reconfigurable macropixel array (RMP, 17 channels) against an equivalent-area
uniform pixel array (81 channels), with a Gaussian-noise robustness sweep
(σ = 0–0.4). The frozen checkpoints and held-out test split (128 samples per
scheme) are included; one command reproduces the manuscript metrics.

## Environment

Python ≥ 3.9. CUDA is optional (evaluation also runs on CPU). Tested on
Windows 11 with Python 3.9 and PyTorch 1.13 (CUDA 11.7).

```bash
pip install -r requirements.txt
```

## Run

```bash
python evaluate.py                    # defaults: data/, checkpoints/, outputs/
python evaluate.py --device cpu       # force CPU
python evaluate.py --output-dir outputs/run1
```

Writes to `outputs/`: `clean_accuracy.csv`, `noise_robustness.csv` (mean ± std
per σ, both tasks) and `noise_robustness.png`.

## Expected results

Deterministic with the default seed: RMP direction 89.1% (uniform 49.2%, chance
level); letter accuracy 99.8% / 90.1% at σ = 0.2 / 0.4 (uniform 93.2% / 76.3%).

## License

MIT — see `LICENSE`.
