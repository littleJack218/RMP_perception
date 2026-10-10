"""RMP spatio-temporal event encoding (the trained layout).

The released checkpoint was trained on event tensors built by the RMP
acquisition protocol: each sample covers one CIP exposure window and its
events are accumulated into 1-ms integration bins over the two halves of the
window, relative to the mid-exposure anchor ``t*``:

    channels [0, 2n)      post-anchor half, one (positive, negative) plane
                          pair per 1-ms interval
    channels [2n, 4n)     pre-anchor half, same pairs but with the polarity
                          order swapped, so the pair difference carries the
                          sign flip the reconstruction relies on

with ``n = n_bins = 117`` (1-ms bins over the 117-ms tensor width) and counts
clipped into the outermost bin when a millisecond lies outside the tensor.

Pair layout (this is what the checkpoint's ``enc_noisy.interval_weight`` of
length ``2*n`` expects):

    channel 2k     post half, interval k, positive-polarity count
    channel 2k + 1 post half, interval k, negative-polarity count
    channel 2n+2k  pre half,  interval k, negative-polarity count
    channel 2n+2k+1 pre half, interval k, positive-polarity count

Interval k of the post half collects the events with
``t - t* = k ms`` (k >= 1); the pre half collects ``t* - t = k ms`` (k >= 0,
which is where an event exactly at the anchor lands). Timestamps are integer
milliseconds, so the binning is exact and reproducible.
"""
import numpy as np

N_BINS = 117            # 1-ms bins per half of the exposure window
EXPOSURE_MS = 116.0     # nominal window width; the tensor is n_bins wide
H, W = 180, 320         # CIP frame resolution (LR domain)


# ---------------------------------------------------------------- event files
def load_event_txt(path):
    """Read an ``t x y p`` event file (seconds, column, row, polarity).

    Returns dict with timestamps in integer milliseconds and compact dtypes:
    x is the column (0..319) and needs uint16 - casting it to uint8 wraps
    events into the left 64 columns.
    """
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[None]
    order = np.argsort(raw[:, 0], kind="stable")
    raw = raw[order]
    return _with_ms_histogram({"t": np.rint(raw[:, 0].astype(np.float64) * 1000.0).astype(np.int32),
                               "x": raw[:, 1].astype(np.uint16),
                               "y": raw[:, 2].astype(np.uint8),
                               "p": raw[:, 3].astype(np.uint8)})


def load_esim_timeline(path):
    """Frame timeline of one sequence, in seconds on the event clock.

    Each line is ``<timestamp> <frame>.png``; the 100 source frames sit on
    every 8th entry, so shipped frame ``k`` sits at ``t[offset + 8 k]``.
    """
    return np.array([float(line.split()[0]) for line in open(path)], dtype=np.float64)


def _with_ms_histogram(ev):
    """Attach events-per-millisecond counts, so window bounds are O(1) lookups."""
    ts = ev["t"]
    if len(ts) == 0:
        ev["t_counts"], ev["t_counts_offset"] = np.zeros(1, np.int32), 0
        ev["t_counts_cum"] = np.zeros(2, np.int64)
        return ev
    lo, hi = int(ts[0]), int(ts[-1])
    counts = np.bincount(ts.astype(np.int64) - lo, minlength=hi - lo + 1).astype(np.int32)
    ev["t_counts"] = counts
    ev["t_counts_offset"] = lo
    ev["t_counts_cum"] = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return ev


def _bounds_ms(ev, lo_ms, hi_ms):
    """Event index range for the millisecond window ``[lo_ms, hi_ms]``."""
    counts = ev["t_counts"]
    t0 = int(ev["t_counts_offset"])
    cum = ev["t_counts_cum"]
    n = len(cum) - 1
    k0 = min(max(lo_ms - t0, 0), n)          # clip: a window may lie outside the clip
    k1 = min(max(hi_ms - t0 + 1, 0), n)
    return int(cum[k0]), int(cum[k1])


# ------------------------------------------------------------ window encoding
def event_window_tensor(ev, t_start, t_end, t_anchor, n_bins=N_BINS, hw=(H, W)):
    """Encode one exposure window into the (4 * n_bins, H, W) count tensor.

    ``t_start`` / ``t_end`` / ``t_anchor`` are seconds on the event clock;
    timestamps are compared as integer milliseconds (the hardware readout
    granularity). See the module docstring for the channel layout.
    """
    height, width = hw
    out = np.zeros((4 * n_bins, height * width), dtype=np.float32)
    t = ev["t"]                                             # int32 ms
    lo_ms, hi_ms = int(round(t_start * 1e3)), int(round(t_end * 1e3))
    anchor_ms = int(round(t_anchor * 1e3))
    i0, i1 = _bounds_ms(ev, lo_ms, hi_ms)
    if i0 < i1:
        tw = t[i0:i1]
        pix = ev["y"][i0:i1].astype(np.intp) * width + ev["x"][i0:i1].astype(np.intp)
        pos = ev["p"][i0:i1] >= 1
        # post-anchor half: interval k holds the events k ms after the anchor
        f = int(np.searchsorted(tw, anchor_ms, side="right"))
        if f < len(tw):
            k = np.clip(tw[f:] - anchor_ms, 0, n_bins - 1)
            plane = 2 * k + (~pos[f:]).astype(np.intp)      # 2k = positive, 2k+1 = negative
            np.add.at(out, (plane, pix[f:]), 1.0)
        # pre-anchor half: interval k holds the events k ms before the anchor,
        # with the plane pair polarity-swapped
        if f > 0:
            k = np.clip(anchor_ms - tw[:f], 0, n_bins - 1)
            plane = 2 * n_bins + 2 * k + pos[:f].astype(np.intp)
            np.add.at(out, (plane, pix[:f]), 1.0)
    return out.reshape(4 * n_bins, height, width)


def shuffle_event_maps(maps, rng):
    """Event-content control: shuffle every plane's pixels in place.

    The per-plane event count (density) is preserved while the spatial
    structure the reconstruction reads is destroyed. Returns a new array.
    """
    out = maps.copy()
    flat = out.reshape(out.shape[0], -1)
    for r in range(flat.shape[0]):
        rng.shuffle(flat[r])
    return out
