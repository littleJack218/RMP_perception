# RMP scene reconstruction (Fig. 5)

Event-enhanced 4× blur-robust scene reconstruction: the released network
checkpoint, the evaluation code and three of the 30 evaluation scenes
(95 exposure-window samples each) are included; the bundled scenes reproduce
their per-scene rows of the manuscript's reference table, whose 30-scene
average is 24.9 dB / 0.659 SSIM (vs 21.1 dB / 0.560 conventional).

## Environment

Python ≥ 3.9 (`torch >= 1.13`, `numpy`, `opencv-python`). GPU recommended but
not required (falls back to CPU). Tested on Windows 11 with Python 3.9 and
PyTorch 1.13 (CUDA 11.7).

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/evaluate.py --self-test        # smoke test: strict checkpoint load + forward pass
python src/evaluate.py                    # bundled scenes vs the paper's reference values (~8 min on one RTX 3060)
python src/evaluate.py --scenes 028       # a single scene
python src/evaluate.py --detail           # + dynamic/static pixel and region analyses
python src/bandwidth.py                   # RIP bandwidth of the bundled scenes (Fig. 5g)
```

`results/expected_metrics.json` and `results/per_scene_expected.csv` hold the
paper's reference values (full 30-scene split); `python src/evaluate.py`
prints each bundled-scene value next to its reference.

## Data

The bundled frames derive from the REDS dataset (Nah et al., CVPRW 2019) and
the events from ESIM (Rebecq et al., CoRL 2018); cite the originals when
reusing them. The network core is adapted from the official eSL-Net++
implementation (Yu et al., IEEE TPAMI 2023, MIT license).

## License

MIT — see `LICENSE`. Training code and the full split are not part of this
release (recipe in the manuscript's Methods).
