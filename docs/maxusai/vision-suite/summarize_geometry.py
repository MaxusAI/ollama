#!/usr/bin/env python3
"""Render the geometry sweep (SPEC C13-C18) as a markdown table.

    python3 summarize_geometry.py <tag-prefix> <arm> [<tag-prefix> <arm> ...]

e.g. python3 summarize_geometry.py frm bbox_contract_real_1img

One row per geometry, one column group per model. Exists because ADR 0012 rule 8
forbids transcribing generator output by hand, and the geometry axis has no
template of its own -- T1/T2/T3 are all keyed on model, not on image size.

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


def main():
    args = sys.argv[1:]
    rundir = os.path.dirname(os.path.abspath(__file__))
    if args and args[0] == "--dir":
        rundir, args = args[1], args[2:]
    if len(args) < 2 or len(args) % 2:
        sys.exit(__doc__)

    for prefix, arm in zip(args[::2], args[1::2]):
        rows, models = load(prefix, arm, rundir)
        if not rows:
            print(f"\n**{arm}** — no cells found for prefix {prefix!r}\n")
            continue
        print(f"\n**{arm}** (`{prefix}-*`)\n")
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
