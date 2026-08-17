#!/usr/bin/env python3
"""Aggregate repeated runs of one arm into mean and spread.

ADR 0023 asks for n=3 per think-on cell because on-policy sampling is not
deterministic. Three scores files then have to become one row, and doing that
arithmetic by hand into a table is precisely what ADR 0012 rule 8 forbids —
"never transcribe generator output by hand". So this does it.

    python3 summarize_reps.py rocm-n3-qwen38-thinkoff rocm-n3-qwen38-thinkon

Tags are arm names. Files are matched as scores_<tag>.json and
scores_<tag>-rep<N>.json, so an arm measured once and an arm measured three
times are both handled, and the reported n says which.

Spread is max-min, not a standard deviation: with n=3 the range is what a reader
can actually reason about, and it is the number that tells you whether a
difference between two arms survives the noise within one arm.
"""
import glob
import json
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

# (section, key, label, kind). "bool" is scored as a pass count, not a mean.
METRICS = [
    ("scene_single", "bbox_mean_iou", "scene bbox IoU", "float"),
    ("scene_single", "labels_found", "scene labels", "int"),
    ("scene_single", "colors_right", "scene colors", "int"),
    ("scene_single", "serial_found", "scene serial", "bool"),
    ("document_single", "items_found", "doc items", "int"),
    ("document_single", "qty_price_right", "doc qty+price", "int"),
    ("document_single", "total_right", "doc total", "bool"),
    ("document_single", "name_bbox_mean_iou", "doc name_bbox IoU", "float"),
    ("multi_3img", "q1_right", "multi q1", "bool"),
    ("multi_3img", "q2_right", "multi q2", "bool"),
    ("multi_3img", "q4_bbox_hit", "multi q4-bbox", "bool"),
    ("multi_3img", "chart_values_found", "multi chart", "int"),
    ("finetext", "recall_22px", "finetext 22px", "int"),
    ("finetext", "recall_16px", "finetext 16px", "int"),
    ("finetext", "recall_12px", "finetext 12px", "int"),
    ("finetext", "recall_9px", "finetext 9px", "int"),
    ("finetext", "recall_7px", "finetext 7px", "int"),
    # Derived, and the honest way to read the tiers. Every model observed so far
    # emits all 20 codes regardless of what it can resolve (finetext_probe.py
    # says so), so a zeroed small tier means FABRICATED codes, not omitted ones.
    # Per-tier recall alone reads as "lost the 9px tier" when the tiers can swap
    # between runs while the total correct stays put — that swap is which
    # fabrications happened to coincide with ground truth, not a capability
    # change. Measured 2026-08-17: two think-on runs scored 4/4/4/1/0 and
    # 4/4/4/0/1, identical totals, opposite tiers.
    ("finetext", "_correct", "finetext correct /20", "int"),
    ("finetext", "_fabricated", "finetext fabricated", "int"),
]

TIERS = (22, 16, 12, 9, 7)


def derive(run):
    """Add the derived finetext fields so METRICS can treat them uniformly."""
    ft = run.get("finetext")
    if not ft or "total_found" not in ft:
        return run
    correct = sum(ft.get(f"recall_{t}px", 0) for t in TIERS)
    ft["_correct"] = correct
    ft["_fabricated"] = ft["total_found"] - correct
    return run


def load_arm(tag):
    paths = sorted(set(glob.glob(os.path.join(DIR, f"scores_{tag}.json"))
                       + glob.glob(os.path.join(DIR, f"scores_{tag}-rep*.json"))))
    return [derive(json.load(open(p))) for p in paths], paths


def cell(runs, section, key, kind):
    vals = [r[section][key] for r in runs
            if section in r and key in r.get(section, {})]
    if not vals:
        return "—", None
    if kind == "bool":
        hits = sum(1 for v in vals if v)
        return (f"{hits}/{len(vals)} ✅" if hits == len(vals)
                else f"{hits}/{len(vals)} ❌"), 0 if hits in (0, len(vals)) else 1
    lo, hi = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    spread = hi - lo
    if kind == "float":
        s = f"{mean:.3f}" + (f" ±{spread / 2:.3f}" if spread else "")
    else:
        s = f"{mean:.1f}".rstrip("0").rstrip(".") + (f" [{lo}–{hi}]" if spread else "")
    return s, spread


def main():
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        return 2

    arms = []
    for tag in tags:
        runs, paths = load_arm(tag)
        if not runs:
            print(f"# no scores files for {tag}", file=sys.stderr)
            continue
        arms.append((tag, runs))
        print(f"# {tag}: n={len(runs)}  " + ", ".join(os.path.basename(p) for p in paths),
              file=sys.stderr)

    print("| metric | " + " | ".join(f"{t} (n={len(r)})" for t, r in arms) + " |")
    print("|---" * (len(arms) + 1) + "|")
    for section, key, label, kind in METRICS:
        cells = [cell(runs, section, key, kind)[0] for _, runs in arms]
        if all(c == "—" for c in cells):
            continue
        print(f"| {label} | " + " | ".join(cells) + " |")

    # The point of repeating: how much does one arm move on its own?
    print()
    print("Within-arm spread (max-min), the bar any cross-arm claim must clear:")
    for tag, runs in arms:
        if len(runs) < 2:
            print(f"  {tag}: n={len(runs)}, no spread measurable")
            continue
        worst = []
        for section, key, label, kind in METRICS:
            if kind == "bool":
                continue
            _, sp = cell(runs, section, key, kind)
            if sp:
                worst.append((sp, label))
        worst.sort(reverse=True)
        if worst:
            print(f"  {tag}: " + ", ".join(f"{l} {s:g}" for s, l in worst[:4]))
        else:
            print(f"  {tag}: identical across all {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
