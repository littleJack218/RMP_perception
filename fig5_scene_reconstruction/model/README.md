# Released checkpoint

`rmp_real_ft4_latest_G.pth` — the reconstruction network of Fig. 5, a plain
state dict of the 19 tensors of `src/network.py` (1,403,159 parameters, 5.6 MB):

```python
from network import Net, load_checkpoint
net = Net(n_bins=117)
load_checkpoint(net, "model/rmp_real_ft4_latest_G.pth")
```

`python src/evaluate.py --self-test` loads this file strictly and runs a
forward pass as a smoke test. Training recipe: manuscript Methods.
