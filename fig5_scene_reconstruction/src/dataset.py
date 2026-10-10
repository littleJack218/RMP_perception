"""Bundled demonstration scenes: CIP frames, sharp ground truth, event windows.

The package ships three of the 30 evaluation scenes of the paper (028, 012 and
013) in their native acquisition layout:

    testdata/rmp_root/Val_lr/{seq}_{n:08d}.png   blurred CIP frame, 180x320
    testdata/rmp_root/Val_hr/{seq}_{n:08d}.png   sharp frame, 720x1280 (GT)
    testdata/events/{seq}.txt                    ESIM event stream, t x y p
    testdata/esim/val_esim/image/{seq}.txt       frame timeline on the event clock

One sample is one exposure window: frame ``n`` of ``seq``. Its CIP input is the
blurred frame ``n``, its target is the sharp frame ``n`` (the first frame of
the window, the paper's protocol), and its RIP input is the event tensor of the
window ``[t(n), t(n+2)]`` anchored at its midpoint - the timestamps come from
the shipped frame timeline, where the source frames sit on every 8th entry.
Samples exist for every ``n`` whose frames ``n`` and ``n + 1`` are shipped
(95 per scene, 285 in total).
"""
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data

from encoding import event_window_tensor, load_esim_timeline, load_event_txt

ESIM_ANCHOR_STEP = 8        # source frames sit on every 8th timeline entry
ESIM_ANCHOR_OFFSET = 1      # first shipped source frame


def imread_gray(path):
    """Read an 8-bit grayscale PNG (path-safe: cv2.imread fails on non-ASCII paths)."""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


class BundledScenes(data.Dataset):
    def __init__(self, root, n_bins=117, cache_seqs=3):
        self.root = root
        self.n_bins = n_bins
        self.hr_dir = os.path.join(root, "rmp_root", "Val_hr")
        self.blur_dir = os.path.join(root, "rmp_root", "Val_lr")
        self.event_dir = os.path.join(root, "events")
        self.esim_dir = os.path.join(root, "esim", "val_esim", "image")
        self._cache_seqs = cache_seqs
        self._events = {}
        self._timelines = {}
        self.samples = self._index()
        self.seqs = sorted({s["seq"] for s in self.samples})

    # ------------------------------------------------------------------ index
    def _index(self):
        samples = []
        for name in sorted(os.listdir(self.blur_dir)):
            if not name.endswith(".png") or "_" not in name:
                continue
            seq, idx = name[:-4].split("_", 1)
            n = int(idx)
            hr0 = os.path.join(self.hr_dir, "%s_%08d.png" % (seq, n))
            hr1 = os.path.join(self.hr_dir, "%s_%08d.png" % (seq, n + 1))
            if not (os.path.exists(hr0) and os.path.exists(hr1)):
                continue
            if not os.path.exists(os.path.join(self.event_dir, seq + ".txt")):
                continue
            samples.append({"seq": seq, "idx": n,
                            "blur": os.path.join(self.blur_dir, name), "hr0": hr0})
        return samples

    def __len__(self):
        return len(self.samples)

    # ------------------------------------------------------------------- data
    def _events_of(self, seq):
        ev = self._events.pop(seq, None)
        if ev is None:
            ev = load_event_txt(os.path.join(self.event_dir, seq + ".txt"))
        self._events[seq] = ev                      # keep the most recent few in memory
        while len(self._events) > self._cache_seqs:
            self._events.pop(next(iter(self._events)))
        return ev

    def _timeline_of(self, seq):
        tl = self._timelines.get(seq)
        if tl is None:
            tl = load_esim_timeline(os.path.join(self.esim_dir, seq + ".txt"))
            self._timelines[seq] = tl
        return tl

    def event_window(self, seq, n):
        """(4 * n_bins, H, W) event-count tensor of the exposure window of frame n."""
        tl = self._timeline_of(seq)
        j0 = ESIM_ANCHOR_OFFSET + ESIM_ANCHOR_STEP * n
        t_start = float(tl[min(max(j0, 0), len(tl) - 1)])
        t_end = float(tl[min(max(j0 + 2 * ESIM_ANCHOR_STEP, 0), len(tl) - 1)])
        return event_window_tensor(self._events_of(seq), t_start, t_end,
                                   0.5 * (t_start + t_end), n_bins=self.n_bins)

    def __getitem__(self, i):
        s = self.samples[i]
        blur = imread_gray(s["blur"])
        if blur is None:
            raise RuntimeError("failed to read %s" % s["blur"])
        gt = imread_gray(s["hr0"])
        if gt is None:
            raise RuntimeError("failed to read %s" % s["hr0"])
        return {"blur": np.ascontiguousarray(blur[None].astype(np.float32) / 255.0),
                "gt": np.ascontiguousarray(gt),
                "events": self.event_window(s["seq"], s["idx"]),
                "seq": s["seq"], "idx": s["idx"]}
