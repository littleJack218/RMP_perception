"""RIP transmission bandwidth of the bundled scenes (Fig. 5g).

The RIP output is sparse: only pixels whose luminance changes emit, and each
event costs 10 bit. The scene bandwidth is therefore the mean event rate of its
clip times 10 bit, and the RMP output of the scene is that RIP stream next to
the CIP frames it accompanies:

    CIP          180 x 320 x 8 bit @ 30 Hz          =    13.824 Mbps
    conventional 720 x 1280 x 8 bit @ 120 Hz        =   884.736 Mbps
    RMP total    CIP + RIP(measured)                =   884.736 / total  compression

The manuscript's Fig. 5g quotes the demonstration scene (028) at 3.65 Mbps RIP
and 51x compression, and a per-scene range of 0.59-9.20 Mbps (median 5.25)
across the 30-scene evaluation split.

Usage
    python src/bandwidth.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIT_PER_EVENT = 10.0
CIP_MBPS = 180 * 320 * 8 * 30 / 1e6                    # 13.824
CONV_MBPS = 720 * 1280 * 8 * 120 / 1e6                 # 884.736
REFERENCE = {"028": "demonstration scene of Fig. 5g (paper: 3.65 Mbps, 51x)"}


def timestamps(path):
    """First column (seconds) of an event file."""
    with open(path) as fh:
        return np.array([float(line.split(maxsplit=1)[0]) for line in fh], dtype=np.float64)


def main():
    root = os.path.join(HERE, "testdata", "events")
    print("%-6s %-12s %-12s %-14s %-16s" % ("scene", "events", "span (s)", "RIP (Mbps)", "RMP total"))
    for name in sorted(os.listdir(root)):
        if not name.endswith(".txt"):
            continue
        seq = name[:-4]
        t = timestamps(os.path.join(root, name))
        span = float(t.max() - t.min())
        rip = len(t) / span * BIT_PER_EVENT / 1e6
        total = CIP_MBPS + rip
        note = REFERENCE.get(seq, "")
        print("%-6s %-12d %-12.3f %-14.3f %-40s"
              % (seq, len(t), span, rip,
                 "%.3f Mbps (%.1fx)  %s" % (total, CONV_MBPS / total, note)))
    print("\nconstants: CIP %.3f Mbps | conventional %.3f Mbps | 10 bit per event"
          % (CIP_MBPS, CONV_MBPS))
    print("the 30-scene split spans 0.59-9.20 Mbps (median 5.25); scene 028 is the")
    print("demonstration scene of Fig. 5g (3.65 Mbps, 51x compression, 2.0% of conventional)")


if __name__ == "__main__":
    main()
