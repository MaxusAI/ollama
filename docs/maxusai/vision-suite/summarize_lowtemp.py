#!/usr/bin/env python3
"""Render the sampling-comparison arm as (score, num_ctx, num_predict) per model.

ADR 0012 rule 8: a result is the answer AND what it cost to obtain. Cap counts
are harness diagnostics, not results, so they do not appear as a headline here —
a cell that never terminated is reported as a MISSING SCORE at a stated window,
which is what it is.

This arm is exploratory (ADR 0012 rule 7), so it is exempt from the T1/T2 shapes
but still carries the provenance header and marks every invalid cell explicitly.

Usage: summarize_lowtemp.py <tag-prefix> [<tag-prefix> ...]
       summarize_lowtemp.py lt          # the TEMPERATURE=0.01 arm
"""
import json
import os
import re
import sys
import glob

DIR = os.path.dirname(os.path.abspath(__file__))
ARMS = ["bbox_contract", "bbox_contract_multi", "bbox_contract_reasoning",
        "bbox_contract_pinned", "bbox_contract_perobject", "bbox_contract_anchored",
        "bbox_contract_adv_real", "bbox_contract_adv_norm1"]
ENGINE = lambda m: "**MLX**" if re.search(r"nvfp4|mxfp8|mlx-bf16", m) else "GGUF"


def pretty(tag):
    """Score-file tags mangle both ':' and '.' to '_', so 'qwen3_6_35b-a3b' is
    ambiguous on a naive split. Restore the known families explicitly."""
    for fam in ("qwen3.6", "qwen3.8", "qwen3.5", "gemma4", "nemotron3"):
        mangled = fam.replace(".", "_") + "_"
        if tag.startswith(mangled):
            return f"{fam}:{tag[len(mangled):]}"
    return tag.replace("_", ":", 1)


def collect(prefix):
    """{(model, mode): [per-repeat dict]}"""
    out = {}
    for p in sorted(glob.glob(f"{DIR}/scores_{prefix}*_think*.json")):
        base = os.path.basename(p)[7:-5]
        m = re.match(rf"{prefix}(\d+)_(.+)_think(false|on)$", base)
        if not m:
            continue
        out.setdefault((m.group(2), m.group(3)), []).append(json.load(open(p)))
    return out


def main():
    prefixes = sys.argv[1:] or ["lt"]
    for prefix in prefixes:
        data = collect(prefix)
        if not data:
            print(f"no scores for prefix {prefix!r}")
            continue
        print("| Model | Engine | think | contract_followed | mean IoU (scored cells) | no result | num_ctx | num_predict |")
        print("|---|---|---|---|---|---|---|---|")
        for (mdl, mode) in sorted(data):
            reps = data[(mdl, mode)]
            foll, iou, missing, ctx, npd = [], [], [], set(), set()
            for d in reps:
                f = n = 0
                ious = []
                for a in ARMS:
                    c = d.get(a)
                    if not c:
                        continue
                    ctx.add(c.get("req_num_ctx") or c.get("num_ctx"))
                    npd.add(c.get("req_num_predict") or c.get("num_predict"))
                    # A cell whose generation never terminated has no score to
                    # report. Say so; do not score it as a zero.
                    if c.get("eval_count") is not None and c.get("eval_count") == c.get("num_predict"):
                        n += 1
                        continue
                    f += bool(c.get("contract_followed"))
                    if c.get("iou_declared"):
                        ious.append(c["iou_declared"])
                foll.append(f)
                missing.append(n)
                iou.append(round(sum(ious) / len(ious), 3) if ious else None)
            fmt = lambda xs: "/".join("—" if x is None else str(x) for x in xs)
            one = lambda s: str(sorted(s)[0]) if len(s) == 1 else "/".join(map(str, sorted(s)))
            print(f"| {pretty(mdl)} | {ENGINE(mdl)} | {mode} | {fmt(foll)} of 8 | "
                  f"{fmt(iou)} | {fmt(missing)} of 8 | {one(ctx)} | {one(npd)} |")
        print()
        print("Three values per cell = three repeats. `no result` counts cells whose")
        print("generation never terminated at the stated window, so they carry no score;")
        print("they are excluded from the IoU mean rather than scored as zero.")


if __name__ == "__main__":
    main()
