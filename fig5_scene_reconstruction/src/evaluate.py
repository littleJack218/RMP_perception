"""Reproduce the scene-reconstruction results of the RMP manuscript (Fig. 5).

Runs the released checkpoint over the bundled demonstration scenes and prints,
next to each reproduced value, the reference value of the paper:

    scene   events   zero   shuffled   conventional   delta

* ``events``       RMP reconstruction from the exposure-window event tensor
* ``zero``         same run with the event tensor zeroed (no-event control)
* ``shuffled``     density-preserving spatial shuffle of every event plane
                   (content control: the timing/structure is destroyed)
* ``conventional`` bicubic upsampling of the blurred CIP frame, both at
                   720x1280 - the conventional-imaging baseline
* ``delta``        events - conventional, the reconstruction gain of Fig. 5e

Three scenes of the 30-scene evaluation split ship with this package (028:
highest mean gain of the split; 012: highest single-tile gain, the scene of the
manuscript's qualitative panels; 013: low-motion control), 95 samples each.
The full-split averages of the paper are printed as a reference block; they
follow from the per-scene values in ``results/per_scene_expected.csv`` (30
scenes, 2,850 samples), which the bundled scenes cross-check sample by sample.

Usage
    python src/evaluate.py --self-test                    # no data needed
    python src/evaluate.py                                # bundled scenes
    python src/evaluate.py --scenes 012 --detail          # one scene, + extras
    python src/evaluate.py --out-json results/demo_run.json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import BundledScenes                                  # noqa: E402
from metrics import masked_psnr, psnr8, ssim                        # noqa: E402
from network import Net, load_checkpoint                            # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILE = 128
HR_HW = (720, 1280)

# Reference values of the paper (full 30-scene split, 2,850 samples).
FULL_SPLIT = {
    "rmp": (24.86, 0.659),
    "zero_events": (21.02, 0.585),
    "shuffled_events": (19.31, 0.465),
    "conventional": (21.08, 0.560),
    "vs_conventional_db": 3.78,
}


def to_u8(x):
    return np.clip(np.round(np.asarray(x) * 255.0), 0, 255).astype(np.uint8)


def bicubic(blur_u8):
    return cv2.resize(blur_u8, (HR_HW[1], HR_HW[0]), interpolation=cv2.INTER_CUBIC)


def dynamic_mask(event_maps, min_events=2, dilate=5):
    """Dynamic-pixel mask: pixels with >= min_events in the window, 5x5 dilated.

    Defined purely on the input side and resized (nearest) to the reconstruction
    resolution; never derived from reconstruction results.
    """
    counts = event_maps.sum(axis=0)
    mask = (counts >= min_events).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8)).astype(bool)
    return cv2.resize(mask.astype(np.uint8), (HR_HW[1], HR_HW[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def region_gains(pred_u8, gt_u8, bic_u8, tile=TILE):
    """Blurred-region and best-tile gains of `pred` over the conventional baseline.

    The blurred region is the worst quarter of the 128x128 tiles by baseline
    error; the best tile is the largest gain over all tiles.
    """
    errs = []
    for i in range(0, gt_u8.shape[0] - tile + 1, tile):
        for j in range(0, gt_u8.shape[1] - tile + 1, tile):
            s = (slice(i, i + tile), slice(j, j + tile))
            errs.append((((bic_u8[s].astype(np.float64) - gt_u8[s]) ** 2).mean(), s))
    k = max(1, len(errs) // 4)
    blurred = [s for _, s in sorted(errs, reverse=True)[:k]]
    reg = [psnr8(pred_u8[s], gt_u8[s]) - psnr8(bic_u8[s], gt_u8[s]) for s in blurred]
    best = max(psnr8(pred_u8[s], gt_u8[s]) - psnr8(bic_u8[s], gt_u8[s]) for _, s in errs)
    return float(np.mean(reg)), float(best)


def run(args):
    device = torch.device(args.device)
    net = Net(n_bins=args.n_bins).to(device).eval()
    load_checkpoint(net, args.checkpoint, device=device)
    print("checkpoint loaded: %s | parameters %s"
          % (os.path.basename(args.checkpoint), "{:,}".format(
              sum(p.numel() for p in net.parameters()))))

    ds = BundledScenes(args.data_dir, n_bins=args.n_bins)
    keep = [s for s in ds.seqs if args.scenes == "all" or s in set(args.scenes.split(","))]
    idx = [i for i, s in enumerate(ds.samples) if s["seq"] in keep]
    if args.stride > 1:
        idx = idx[:: args.stride]
    rng = np.random.default_rng(args.seed)
    print("evaluating %d samples of scene(s) %s on %s\n"
          % (len(idx), ",".join(keep), device))

    acc, per_sample, t0 = {}, [], time.time()
    for n, i in enumerate(idx):
        s = ds[i]
        blur_u8 = to_u8(s["blur"][0])
        gt_u8 = s["gt"]
        blur = torch.from_numpy(s["blur"][None]).to(device)
        maps = s["events"]
        maps_zero = np.zeros_like(maps)
        maps_shuf = shuffle_planes(maps, rng)

        row = {"seq": s["seq"], "idx": s["idx"]}
        outs = {}
        with torch.no_grad():
            for tag, m in (("ev", maps), ("zero", maps_zero), ("shuf", maps_shuf)):
                t = torch.from_numpy(m[None]).to(device)
                outs[tag] = to_u8(net(blur, t)[0, 0].float().cpu().numpy())
        outs["bic"] = bicubic(blur_u8)
        for tag in ("ev", "zero", "shuf", "bic"):
            row["psnr_" + tag] = psnr8(outs[tag], gt_u8)
            row["ssim_" + tag] = ssim(outs[tag], gt_u8)
            acc.setdefault(tag, []).append(row["psnr_" + tag])
            acc.setdefault(tag + "_ssim", []).append(row["ssim_" + tag])
        if args.detail:
            mask = dynamic_mask(maps)
            row["psnr_dyn"] = masked_psnr(outs["ev"], gt_u8, mask)
            row["psnr_dyn_bic"] = masked_psnr(outs["bic"], gt_u8, mask)
            row["psnr_sta"] = masked_psnr(outs["ev"], gt_u8, ~mask)
            row["psnr_sta_bic"] = masked_psnr(outs["bic"], gt_u8, ~mask)
            row["cover"] = float(mask.mean())
            row["reg_mean"], row["tile_best"] = region_gains(outs["ev"], gt_u8, outs["bic"])
        per_sample.append(row)
        if (n + 1) % 20 == 0:
            print("  ... %d/%d  (%.0f s)" % (n + 1, len(idx), time.time() - t0))

    # ---------------------------------------------------------------- report
    def scene_mean(seq, key):
        v = [r[key] for r in per_sample if r["seq"] == seq]
        return float(np.mean(v))

    print("\n" + "=" * 78)
    print("BUNDLED SCENES  (PSNR dB; reference = full-split per-scene value of the paper)")
    print("=" * 78)
    ref = load_reference(os.path.join(HERE, "results", "expected_metrics.json"))
    ref_scenes = ref.get("bundled_scenes", {}).get("scenes", {})
    print("%-6s %-20s %-20s %-12s" % ("scene", "reproduced", "reference (paper)", "n"))
    for seq in keep:
        ev, bic = scene_mean(seq, "psnr_ev"), scene_mean(seq, "psnr_bic")
        r = ref_scenes.get(seq, {})
        ref_txt = ("%.2f / %.2f (+%.2f)" % (r["rmp_psnr"], r["conv_psnr"],
                                            r["rmp_psnr"] - r["conv_psnr"])
                   if r else "-")
        n = sum(1 for r_ in per_sample if r_["seq"] == seq)
        n_txt = "%d/%d" % (n, r["n_samples"]) if r and n < r["n_samples"] else str(n)
        if r and n < r["n_samples"]:
            ref_txt += " (subset)"
        print("%-6s %-20s %-20s %-12s"
              % (seq, "%.2f / %.2f (+%.2f)" % (ev, bic, ev - bic), ref_txt, n_txt))
    m = {k: float(np.mean(v)) for k, v in acc.items()}
    have_ref = all(s in ref_scenes for s in keep)
    ref_mean = ("%.2f / %.2f (+%.2f)" % (
        np.mean([ref_scenes[s]["rmp_psnr"] for s in keep]),
        np.mean([ref_scenes[s]["conv_psnr"] for s in keep]),
        np.mean([ref_scenes[s]["rmp_psnr"] - ref_scenes[s]["conv_psnr"] for s in keep])
    ) if have_ref else "-")
    print("%-6s %-20s %-20s %-12d"
          % ("MEAN", "%.2f / %.2f (+%.2f)" % (m["ev"], m["bic"], m["ev"] - m["bic"]),
             ref_mean, len(per_sample)))
    print("\ncontrol columns (bundled scenes, mean over %d samples):" % len(per_sample))
    print("  events %.2f / %.3f   zero-events %.2f / %.3f   shuffled %.2f / %.3f"
          % (m["ev"], m["ev_ssim"], m["zero"], m["zero_ssim"], m["shuf"], m["shuf_ssim"]))
    print("  event contribution (events - zero) : %+.2f dB" % (m["ev"] - m["zero"]))
    print("  shuffled below zero-events         : %+.2f dB" % (m["shuf"] - m["zero"]))

    print("\n" + "=" * 78)
    print("FULL-SPLIT REFERENCE OF THE PAPER  (30 scenes, 2,850 samples)")
    print("=" * 78)
    for name, (p, s) in (("RMP reconstruction", FULL_SPLIT["rmp"]),
                         ("zero events", FULL_SPLIT["zero_events"]),
                         ("shuffled events", FULL_SPLIT["shuffled_events"]),
                         ("conventional (bicubic)", FULL_SPLIT["conventional"])):
        print("  %-24s %.2f dB / %.3f SSIM" % (name, p, s))
    print("  gain over conventional   : +%.2f dB" % FULL_SPLIT["vs_conventional_db"])
    print("  (the bundled scenes reproduce the per-scene rows of\n"
          "   results/per_scene_expected.csv, which average to these values)")

    if args.detail:
        print("\nDETAIL over %d samples:" % len(per_sample))
        print("  dynamic pixels : RMP %.2f vs conventional %.2f dB  (%+.2f)"
              % (np.mean([r["psnr_dyn"] for r in per_sample]),
                 np.mean([r["psnr_dyn_bic"] for r in per_sample]),
                 np.mean([r["psnr_dyn"] - r["psnr_dyn_bic"] for r in per_sample])))
        print("  static pixels  : RMP %.2f vs conventional %.2f dB  (%+.2f)   [paper +4.8 / +1.7]"
              % (np.mean([r["psnr_sta"] for r in per_sample]),
                 np.mean([r["psnr_sta_bic"] for r in per_sample]),
                 np.mean([r["psnr_sta"] - r["psnr_sta_bic"] for r in per_sample])))
        print("  dynamic coverage: %.1f%%   [paper 42.9%%]"
              % (100 * np.mean([r["cover"] for r in per_sample])))
        print("  blurred-region gain (worst 25%% of 128x128 tiles): %+.2f dB   [paper +4.51]"
              % np.mean([r["reg_mean"] for r in per_sample]))
        print("  best tile gain per frame: mean %.2f dB, max %.2f dB   [paper 9.07 / 18.85]"
              % (np.mean([r["tile_best"] for r in per_sample]),
                 max(r["tile_best"] for r in per_sample)))

    demo = next((r for r in per_sample if r["seq"] == "012" and r["idx"] == 20), None)
    if demo is not None:
        print("\nDEMONSTRATION FRAME of the manuscript (scene 012, frame 20, Fig. 5c,d):")
        print("  PSNR  %.1f -> %.1f dB (%+.1f)   [paper 17.2 -> 22.8, +5.7]"
              % (demo["psnr_bic"], demo["psnr_ev"], demo["psnr_ev"] - demo["psnr_bic"]))
        print("  SSIM  %.2f -> %.2f           [paper 0.53 -> 0.67]"
              % (demo["ssim_bic"], demo["ssim_ev"]))

    if args.out_json:
        cfg = dict(vars(args))
        for k in ("data_dir", "checkpoint", "out_json"):     # keep local paths out of the dump
            if isinstance(cfg.get(k), str):
                cfg[k] = os.path.basename(cfg[k])
        with open(args.out_json, "w", encoding="utf-8") as fh:
            json.dump({"config": cfg,
                       "mean": {k: float(v) for k, v in m.items()},
                       "per_sample": per_sample}, fh, indent=1)
        print("\nresults written to %s" % args.out_json)


def shuffle_planes(maps, rng):
    """Density-preserving spatial shuffle of every event plane (content control)."""
    out = maps.copy()
    flat = out.reshape(out.shape[0], -1)
    for r in range(flat.shape[0]):
        rng.shuffle(flat[r])
    return out


def load_reference(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def self_test(args):
    net = Net(n_bins=args.n_bins)
    load_checkpoint(net, args.checkpoint, device="cpu")
    net.eval()
    n = 40
    ev = np.zeros((4 * args.n_bins, 180, 320), dtype=np.float32)
    ev[: n, 40:60, 60:80] = 1.0
    with torch.no_grad():
        out = net(torch.rand(1, 1, 180, 320), torch.from_numpy(ev[None]))
    assert out.shape == (1, 1, 720, 1280), out.shape
    assert torch.isfinite(out).all()
    print("self-test PASSED: checkpoint loaded strictly | forward %s | range [%.3f, %.3f]"
          % (tuple(out.shape), float(out.min()), float(out.max())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "testdata"),
                    help="directory with the bundled scenes")
    ap.add_argument("--checkpoint", default=os.path.join(HERE, "model",
                                                         "rmp_real_ft4_latest_G.pth"))
    ap.add_argument("--scenes", default="all", help="'all' or comma-separated scene ids")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every N-th sample")
    ap.add_argument("--n-bins", type=int, default=117)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0, help="shuffled-control seed")
    ap.add_argument("--detail", action="store_true",
                    help="also compute the dynamic/static and region analyses")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    self_test(args) if args.self_test else run(args)


if __name__ == "__main__":
    main()
