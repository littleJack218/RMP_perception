# RMP mixed-timescale action recognition (Fig. 4)

Inference code and data for the action-recognition results (Fig. 4): the
frozen checkpoints and the held-out test split (112 samples) are included;
one command reproduces the paper's metrics (97.3% mixed-scale vs 86.6%
uniform single-scale).

## Environment

Python ≥ 3.9 (`torch`, `numpy`). CPU or GPU both work; evaluation takes
seconds on CPU. Tested on Windows 11 with Python 3.9 and PyTorch 1.13
(CUDA 11.7).

```bash
pip install -r requirements.txt
```

## Run

```bash
python run_inference.py              # paper metrics + per-class table
python verify_s24.py                 # per-sample predictions and the confusion
                                     # matrices of Supplementary Fig. S24
```

## License

MIT — see `LICENSE`. The packaged test data derives from the Weizmann action
dataset and is redistributed for academic reproducibility under its
academic-use terms.
