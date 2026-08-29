#!/usr/bin/env python3
"""Render the geometry sweep (SPEC C13-C18) as a markdown table.

    python3 summarize_geometry.py <tag-prefix> <arm> [<tag-prefix> <arm> ...]

e.g. python3 summarize_geometry.py frm bbox_contract_real_1img

One row per geometry, one column group per model. Template T4 (ADR 0012,
amended 2026-08-20): T1/T2/T3 are all keyed on model, not on image size, so the
geometry axis gets its own shape, and rule 8 forbids transcribing generator
output by hand.

The columns are chosen to make the two failure modes distinguishable at a glance:

  * anchor_ref / ratio -- what frame the model claims, against what was sent.
    Ratio < 1 means it reports a frame SMALLER than the image (SPEC C17).
  * chk / anc / bf -- self_check, hits_anchor, hits_bestfit. The signature that
    matters is `chk=T` with `anc` well below 6: C7 passed an anchor that does
    not actually convert, which is the silent failure C7's amendment describes.
    `bf` above `anc` says a dialect search would have done better than the
    anchor -- diagnostic only per C9, never a consumer path.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_engine_compare import was_capped  # noqa: E402  (SPEC H5)


def pooled(got):
    """Blocks eligible for a pooled mean. A capped cell is an unfinished
    measurement (ADR 0012 conv 9): its eval_count is the harness cap and any
    req/h or IoU derived from it is a setting, not a result. This file
    published '**N req/h** mean serial' with no capped guard at all."""
    return [s for s in got if s and not was_capped(s)]

GEOMS = ["hd", "hd_al32", "hd_al48", "sq320", "vga", "portrait", "uhd", "uhd4k",
         "paste1", "paste2", "paste3", "paste4", "paste5", "paste6"]


def load(prefix, arm, rundir):
    rows, models = {}, []
    for f in sorted(glob.glob(os.path.join(rundir, f"scores_{prefix}-*.json"))):
        m = re.match(rf"scores_{re.escape(prefix)}-(.+?)-\d+_(.+?)_think(\w+)\.json",
                     os.path.basename(f))
        if not m:
            continue
        geom, model, mode = m.groups()
        blk = json.load(open(f)).get(arm)
        if not blk:
            continue
        key = f"{model}/{mode}"
        if key not in models:
            models.append(key)
        rows[(geom, key)] = blk
    return rows, models


def cell(s):
    if not s:
        return " — | — | — "
    sz, ref = s.get("image_size"), s.get("anchor_implied_ref")
    typ = s.get("anchor_implied_type")
    if ref and sz and ref[0]:
        frame = f"{ref[0]}×{ref[1]}"
        ratio = f"{ref[0] / sz[0]:.2f}×"
    else:
        frame, ratio = f"*{typ}*", "—"
    chk = "✅" if s.get("self_check") else "❌"
    anc, bf = s.get("hits_anchor", "—"), s.get("hits_bestfit", "—")
    # The silent failure C7's amendment warns about: passed the checks, does not
    # convert. Flagged inline so it cannot be skimmed past.
    flag = " ⚠" if (s.get("self_check") and isinstance(anc, int) and anc < 6) else ""
    return f"{frame} | {ratio} | {chk} {anc}/{bf}{flag}"


def perf_table(rows, models):
    """Cost and accuracy per geometry.

    prompt_eval_count is the column that makes the image-size axis legible: it is
    the IMAGE token cost, and how it moves across geometries is the difference
    between an architecture that budget-fills and one that scales with input.
    Quoting a bbox result without it hides why a 320x320 cell and a 4K cell cost
    what they do.

    IoU is `iou_anchor` -- accuracy AFTER conversion through the anchor-derived
    space, which is the number a consumer actually gets. hits_anchor says the
    boxes landed on the right objects; IoU says how tightly."""
    print("| geometry | sent | " + " | ".join(
        f"prompt tok | eval tok | IoU | tok/s" for _ in models) + " |")
    print("|---|---|" + "---|" * (4 * len(models)))
    for g in GEOMS:
        present = [rows.get((g, k)) for k in models]
        if not any(present):
            continue
        sz = next(s["image_size"] for s in present if s)
        line = f"| `{g}` | {sz[0]}×{sz[1]} |"
        for s in present:
            if not s:
                line += " — | — | — | — |"
                continue
            iou = s.get("iou_anchor")
            line += (f" {s.get('prompt_eval_count','—')} | {s.get('eval_count','—')} | "
                     f"{iou:.3f} | {s.get('gen_tps','—')} |" if iou is not None else
                     f" {s.get('prompt_eval_count','—')} | {s.get('eval_count','—')} | — | "
                     f"{s.get('gen_tps','—')} |")
        print(line)
    print()
    for k in models:
        got = [rows.get((g, k)) for g in GEOMS]
        got = pooled(got)
        pe = [s["prompt_eval_count"] for s in got if s.get("prompt_eval_count")]
        ious = [s["iou_anchor"] for s in got if s.get("iou_anchor") is not None]
        gts = [s["gen_tps"] for s in got if s.get("gen_tps")]
        pts = [s["prefill_tps"] for s in got if s.get("prefill_tps")]
        sreq = [s["eval_count"] / s["gen_tps"] + s["prompt_eval_count"] / s["prefill_tps"]
                for s in got if s.get("gen_tps") and s.get("prefill_tps")
                and s.get("eval_count") and s.get("prompt_eval_count")]
        if not pe:
            continue
        print(f"- **{k}** — prompt tokens {min(pe)}–{max(pe)} "
              f"(spread {max(pe)-min(pe)}), IoU {min(ious):.3f}–{max(ious):.3f} "
              f"(mean {sum(ious)/len(ious):.3f}), gen {min(gts):.0f}–{max(gts):.0f} tok/s, "
              f"prefill {min(pts):.0f}–{max(pts):.0f} tok/s, "
              f"s/req {min(sreq):.1f}–{max(sreq):.1f} → **{3600/(sum(sreq)/len(sreq)):.0f} req/h** mean serial")


def main():
    args = sys.argv[1:]
    rundir = os.path.dirname(os.path.abspath(__file__))
    if args and args[0] == "--dir":
        rundir, args = args[1], args[2:]
    perf = "--perf" in args
    args = [a for a in args if a != "--perf"]
    if len(args) < 2 or len(args) % 2:
        sys.exit(__doc__)

    for prefix, arm in zip(args[::2], args[1::2]):
        rows, models = load(prefix, arm, rundir)
        if not rows:
            print(f"\n**{arm}** — no cells found for prefix {prefix!r}\n")
            continue
        print(f"\n**{arm}** (`{prefix}-*`)\n")
        if perf:
            perf_table(rows, models)
            continue
        head = "| geometry | sent |" + "".join(
            f" frame | ratio | chk anc/bf |" for _ in models)
        print(head)
        print("|---|---|" + "---|" * (3 * len(models)))
        for g in GEOMS:
            present = [rows.get((g, k)) for k in models]
            if not any(present):
                continue
            sz = next(s["image_size"] for s in present if s)
            line = f"| `{g}` | {sz[0]}×{sz[1]} |"
            for s in present:
                line += " " + cell(s) + " |"
            print(line)
        print()
        for k in models:
            got = [rows.get((g, k)) for g in GEOMS]
            got = [s for s in got if s]
            silent = sum(1 for s in got
                         if s.get("self_check") and isinstance(s.get("hits_anchor"), int)
                         and s["hits_anchor"] < 6)
            rejected = sum(1 for s in got if not s.get("self_check"))
            perfect = sum(1 for s in got if s.get("hits_anchor") == 6)
            print(f"- **{k}** — {len(got)} geometries: {perfect} convert 6/6, "
                  f"{rejected} rejected by C7, **{silent} silent failures** "
                  f"(C7 passed, anchor does not convert)")


if __name__ == "__main__":
    main()
