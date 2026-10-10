"""eSL-Net++_rmp: the released reconstruction network (inference path).

Architecture: the ISTA-unrolled sparse-learning core (trans / DIT1 / DI1 /
DIT2 / DET1 / DE1 / DET2 / lam, 15 iterations, step 0.01, 4x PixelShuffle head)
follows Yu et al., "Learning to Super-Resolve Blurry Images with Events",
IEEE TPAMI 45, 10027-10043 (2023), https://github.com/ShinyWang33/eSL-Net-Plusplus.

For the RMP device model the event encoder is replaced: the exposure-window
event tensor (an interleaved sequence of 1-ms positive/negative count-plane
pairs, see encoding.py) is reduced by a learnable per-interval weighted signed
sum, `interval_weight` x `event_scale`, bounded to [-1, 1] - the readout the
RIP subpixels' gate-programmed relaxation provides. The 2-plane event feature
is then lifted to the 32-channel intensity feature space by a 1x1 convolution
(`event_up`), making the coupling E to the sparse iterations linear, hence
information-preserving.

This file is written to reproduce the released checkpoint's evaluation exactly;
it is not the training code (the clean-stream counterpart `enc_clean` is kept
so the checkpoint loads strictly, but inference uses `enc_noisy`, as in the
paper's evaluation).

Inputs
    x     : (B, 1, 180, 320)   blurred CIP frame, float in [0, 1]
    event : (B, 468, 180, 320) exposure-window event tensor (encoding.py)
Returns
    out   : (B, 1, 720, 1280)  reconstructed sharp frame
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

N_ITER = 15         # unrolled sparse-learning iterations
STEP = 0.01         # proximal-gradient step of the recursion


class IntervalEncoder(nn.Module):
    """Learnable weighted signed sum over the interval pairs of one stream.

    Input channels are consecutive (positive, negative) plane pairs; the
    encoder forms the signed event frame of each interval, accumulates it with
    a learned per-interval weight, scales and bounds the result to [-1, 1].
    """

    def __init__(self, n_intervals):
        super().__init__()
        self.n_intervals = n_intervals
        self.interval_weight = Parameter(torch.ones(n_intervals))
        self.event_scale = Parameter(torch.ones(1))

    def forward(self, e):
        b, c, h, w = e.shape
        pair = e.view(b, self.n_intervals, 2, h, w)
        signed = pair[:, :, 0] - pair[:, :, 1]              # (B, n, H, W)
        acc = (signed * self.interval_weight.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        acc = acc * self.event_scale
        return acc.clamp(-1.0, 1.0).expand(-1, 2, -1, -1)


class Net(nn.Module):
    def __init__(self, n_bins=117):
        super().__init__()
        self.n_bins = n_bins
        self.enc_noisy = IntervalEncoder(2 * n_bins)   # evaluation stream
        self.enc_clean = IntervalEncoder(2 * n_bins)   # training counterpart
        self.event_up = nn.Conv2d(2, 32, 1, 1, 0, bias=True)
        self.event_out = nn.Conv2d(32, 1, 1, 1, 0, bias=False)  # training-time aux head

        self.trans = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.trans2 = nn.Conv2d(32, 1, 3, 1, 1, bias=False)
        self.DIT1 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.DI1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.DIT2 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.DET1 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.DE1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.DET2 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.lam = Parameter(torch.zeros(1))
        self.shlu = nn.ReLU(True)
        self.shu1 = nn.Conv2d(128, 512, 3, 1, 1, bias=False)
        self.shu2 = nn.Conv2d(128, 512, 3, 1, 1, bias=False)
        self.endconv = nn.Conv2d(128, 1, 3, 1, 1, bias=False)
        self.ps1 = nn.PixelShuffle(2)
        self.ps2 = nn.PixelShuffle(2)

    def encode_events(self, event):
        """(B, 4n, H, W) counts -> (B, 32, H, W) event feature."""
        return self.event_up(self.enc_noisy(event))

    def forward(self, x, event):
        Y = self.trans(x)
        E = self.encode_events(event)

        y1 = torch.mul(Y, E)
        y1 = self.DIT1(y1)
        alpha = self.shlu(y1)

        i_add = torch.mul(Y, self.DI1(alpha))
        e1 = self.lam * E + i_add
        e1 = self.DET1(e1)
        beta = self.shlu(e1)

        for _ in range(N_ITER):
            en = self.DE1(beta)
            y1 = torch.mul(Y, en)
            y1 = self.DIT1(y1)
            y = self.DI1(alpha)
            y = torch.mul(y, torch.mul(en, en))
            y = self.DIT2(y)
            yr = alpha + STEP * y1 - STEP * y

            In = self.DI1(alpha)
            i_add = torch.mul(Y, In)
            e1 = self.lam * E + i_add
            e1 = self.DET1(e1)
            e = self.DE1(beta)
            e = torch.mul(e, self.lam + torch.mul(In, In))
            e = self.DET2(e)
            er = beta + STEP * e1 - STEP * e

            alpha = self.shlu(yr)
            beta = self.shlu(er)

        out = self.shu1(alpha)
        out = self.ps1(out)
        out = self.shu2(out)
        out = self.ps2(out)
        return self.endconv(out)


def load_checkpoint(net, path, device="cpu"):
    """Load the released weights (a plain state dict, or one nested under 'G')."""
    ck = torch.load(path, map_location=device)
    sd = ck["G"] if isinstance(ck, dict) and "G" in ck else ck
    net.load_state_dict(sd, strict=True)
    return net
