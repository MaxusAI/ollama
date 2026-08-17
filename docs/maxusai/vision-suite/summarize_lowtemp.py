#!/usr/bin/env python3
"""Render a repeated ARM as (score, num_ctx, num_predict) per model.

ADR 0012 rule 8: a result is the answer AND what it cost to obtain. Cap counts
are harness diagnostics, not results, so a cell that never terminated is
reported as a MISSING SCORE at a stated window rather than scored as a zero.

Arms are exploratory (rule 7) and so exempt from the T1/T2 shapes, but they
still carry the provenance columns.

EVERY SHARED HELPER IS IMPORTED FROM summarize_engine_compare, NOT REWRITTEN.
The first draft of this file re-implemented engine detection, the capped test,
num_ctx extraction and tag parsing — all four already existed there, and the
re-implementations were subtly different: the local capped test used `==` where
was_capped uses `>=`, so it would have missed a cell that overran its cap.

Usage: summarize_lowtemp.py <tag-prefix> [<tag-prefix> ...]
"""
import glob
import os
import re
import sys

from summarize_engine_compare import (ctx_for, engine_for, load, tag_for,
                                      was_capped)

DIR = os.path.dirname(os.path.abspath(__file__))
ARMS = ["bbox_contract", "bbox_contract_multi", "bbox_contract_reasoning",
        "bbox_contract_pinned", "bbox_contract_perobject", "bbox_contract_anchored",
        "bbox_contract_adv_real", "bbox_contract_adv_norm1"]


def display(tag):
    """Invert tag_for for the families this corpus uses."""
    for fam in ("qwen3.6", "qwen3.8", "qwen3.5", "gemma4", "nemotron3"):
        pre = tag_for(fam) + "_"
        if tag.startswith(pre):
            return f"{fam}:{tag[len(pre):]}"
    return tag


def main():
    for prefix in (sys.argv[1:] or ["lt"]):
        rows = {}
        for p in sorted(glob.glob(f"{DIR}/scores_{prefix}*_think*.json")):
            m = re.match(rf"{prefix}(\d+)_(.+)_think(false|on)$", os.path.basename(p)[7:-5])
            if m:
                rows.setdefault((m.group(2), m.group(3)), []).append(load(p))
        if not rows:
            print(f"no scores for prefix {prefix!r}")
            continue
        print("| Model | Engine | think | contract_followed | mean IoU (scored cells) "
              "| no result | num_ctx | num_predict |")
        print("|---|---|---|---|---|---|---|---|")
        for (tag, mode) in sorted(rows):
            foll, iou, missing, ctxs, npds = [], [], [], [], set()
            for d in rows[(tag, mode)]:
                secs = [d.get(a) for a in ARMS if d.get(a)]
                ctxs.extend(secs)
                npds.update(s.get("num_predict") for s in secs if s.get("num_predict"))
                f = n = 0
                ious = []
                for s in secs:
                    if was_capped(s):        # shared definition, >= not ==
                        n += 1
                        continue
                    f += bool(s.get("contract_followed"))
                    if s.get("iou_declared"):
                        ious.append(s["iou_declared"])
                foll.append(f)
                missing.append(n)
                iou.append(round(sum(ious) / len(ious), 3) if ious else None)
            j = lambda xs: "/".join("—" if x is None else str(x) for x in xs)
            eng = engine_for(display(tag), {})
            npd = "/".join(str(v) for v in sorted(npds)) if npds else "—"
            print(f"| {display(tag)} | {'**MLX**' if eng == 'MLX' else eng} | {mode} "
                  f"| {j(foll)} of 8 | {j(iou)} | {j(missing)} of 8 "
                  f"| {ctx_for(*ctxs)} | {npd} |")
        print()
        print("Three values per cell = three repeats. `no result` counts cells whose")
        print("generation never terminated at the stated window, so they carry no score;")
        print("they are excluded from the IoU mean rather than scored as zero.")


if __name__ == "__main__":
    main()
