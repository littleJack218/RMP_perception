# RMP perception

Code for the recognition and reconstruction experiments of the manuscript
"Retina-inspired reconfigurable macropixels for concurrent static and dynamic
vision at compressed bandwidth". Each folder contains the trained weights and
test data to reproduce one figure with a single command:

| Folder | Figure | Command |
|---|---|---|
| [`fig3_recognition/`](fig3_recognition/) | Fig. 3 (direction 89.1% vs 49.2% chance; letter 90.1% vs 76.3% at noise σ = 0.4) | `python evaluate.py` |
| [`fig4_action_recognition/`](fig4_action_recognition/) | Fig. 4 (97.3% mixed vs 86.6% single-scale) | `python run_inference.py` |
| [`fig5_scene_reconstruction/`](fig5_scene_reconstruction/) | Fig. 5 (24.9 dB / 0.659 SSIM vs 21.1 / 0.560 conventional) | `python src/evaluate.py` |

## Environment

Python ≥ 3.9 with `torch >= 1.13`, `numpy` and `opencv-python`. In each folder:

```bash
pip install -r requirements.txt
```

Developed and tested on Windows 11 with Python 3.9 and PyTorch 1.13 (CUDA 11.7).
CUDA is optional; every evaluation also runs on CPU.

Numerical source data: https://doi.org/10.5281/zenodo.22929192

## License

MIT — see [`LICENSE`](LICENSE). The reconstruction network builds on the
eSL-Net++ architecture (Yu et al., IEEE TPAMI 2023).
