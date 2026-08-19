#!/usr/bin/env python3
"""Render the composable bbox matrix as MARGINAL EFFECTS, not 1,120 rows.

    python3 summarize_matrix.py [--dir D] [--metric hits_declared] <tag-prefix>

The matrix is 8 arms (pin × anchor × coords) × think × geometry × model. Printed
flat that is 1,120 rows nobody reads. What the design is FOR is the effect of each
factor, so this pools the other axes and reports the marginal.

**The pooling is the dangerous part and it is done explicitly.** A marginal that
averages over a factor which INTERACTS with the one under test is a lie with a
number attached. So every marginal is reported per model and per think mode, and
only geometry is pooled — because geometry is the axis the contract SPEC already
measured as invariant under the pin (111 of 112 cells). If that stops being true
the per-geometry spread column will show it: a wide spread means the pooling is
hiding something and the marginal should not be quoted.

`hits_declared` is the default metric because it is the one C2 exists for: it
counts boxes that land correctly IN THE SPACE THE MODEL SAID IT USED. A model can
score 6/6 on `hits_bestfit` while declaring the wrong convention, and that
difference is the entire subject of the contract.

Cells carry `offered_key` and `field_name`: which positional key was OFFERED and
which came back. They differ by family (gemma4 box_2d, qwen/nemotron bbox_2d), so
the `pos` arms are NOT one workload across models — `prompt_sha` differs and the
per-model split is mandatory, not stylistic.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys

FACTORS = ("pin", "anchor", "coords")
ARM_RE = re.compile(r"^bboxm_(pin|free)_(anc|noanc)_(named|pos)$")


def factors_of(arm):
    m = ARM_RE.match(arm)
    if not m:
        return None
    p, a, c = m.groups()
    return {"pin": p == "pin", "anchor": a == "anc", "coords": c}


def load(rundir, prefix):
    """[(model, mode, geometry, arm, factors, block)]"""
    out = []
    for f in sorted(glob.glob(os.path.join(rundir, f"scores_{prefix}-*.json"))):
        m = re.match(rf"scores_{re.escape(prefix)}-(.+?)-\d+_(.+)_think(\w+)\.json",
                     os.path.basename(f))
        if not m:
            continue
        geom, model, mode = m.groups()
        try:
            data = json.load(open(f))
        except Exception:
            continue
        for arm, blk in data.items():
            fac = factors_of(arm)
            if fac and isinstance(blk, dict) and "error" not in blk:
                out.append((model, mode, geom, arm, fac, blk))
    return out


def val(blk, metric):
    v = blk.get(metric)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v) if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--metric", default="hits_declared")
    a = ap.parse_args()

    rows = load(a.dir, a.prefix)
    if not rows:
        sys.exit(f"no matrix cells found for prefix {a.prefix!r} in {a.dir}")

    models = sorted({r[0] for r in rows})
    modes = sorted({r[1] for r in rows})
    geoms = sorted({r[2] for r in rows})
    print(f"\n**{a.metric}** — {len(rows)} arm-results, "
          f"{len(models)} models × {len(modes)} think modes × {len(geoms)} geometries\n")

    # ---- marginal effect of each factor, per model × mode -------------------
    for factor in FACTORS:
        levels = (["named", "pos"] if factor == "coords" else [True, False])
        lab = {True: "on", False: "off", "named": "named", "pos": "positional"}
        print(f"### factor: {factor}\n")
        print("| model | think | " + " | ".join(f"{lab[l]}" for l in levels)
              + " | Δ | geometry spread |")
        print("|---|---|" + "---|" * (len(levels) + 2))
        for mdl in models:
            for mode in modes:
                cells = [r for r in rows if r[0] == mdl and r[1] == mode]
                means, spreads = [], []
                for lv in levels:
                    vs = [val(b, a.metric) for (_, _, _, _, f, b) in cells
                          if f[factor] == lv and val(b, a.metric) is not None]
                    means.append(statistics.mean(vs) if vs else None)
                    # spread ACROSS GEOMETRIES at this level — the pooling check
                    per_g = {}
                    for (_, _, g, _, f, b) in cells:
                        if f[factor] == lv and val(b, a.metric) is not None:
                            per_g.setdefault(g, []).append(val(b, a.metric))
                    gm = [statistics.mean(v) for v in per_g.values()]
                    spreads.append(max(gm) - min(gm) if len(gm) > 1 else 0.0)
                if not any(m is not None for m in means):
                    continue
                d = (means[0] - means[1]) if all(m is not None for m in means) else None
                sp = max(spreads) if spreads else 0.0
                # A wide geometry spread means pooling hides something.
                warn = " ⚠ pooling suspect" if sp > 1.0 else ""
                fm = lambda v: f"{v:.2f}" if v is not None else "—"
                print(f"| {mdl.replace('_','.')[:22]} | {mode} | "
                      + " | ".join(fm(m) for m in means)
                      + f" | {fm(d)} | {sp:.2f}{warn} |")
        print()

    # ---- the dialect confound, stated rather than pooled over ---------------
    keys = {}
    for (mdl, _, _, arm, fac, blk) in rows:
        if fac["coords"] == "pos":
            keys.setdefault(mdl, set()).add((blk.get("offered_key"), blk.get("field_name")))
    if keys:
        print("### positional dialect — offered vs returned\n")
        print("| model | offered → returned |")
        print("|---|---|")
        for mdl, pairs in sorted(keys.items()):
            s = ", ".join(f"`{o}` → `{r}`" + (" **mismatch**" if o and r and o != r else "")
                          for o, r in sorted(pairs, key=lambda x: str(x)))
            print(f"| {mdl.replace('_','.')[:22]} | {s} |")
        print()

    # ---- workload identity: one prompt_sha per (arm, model) or the run is mixed
    shas = {}
    for (mdl, _, _, arm, _, blk) in rows:
        if blk.get("prompt_sha"):
            shas.setdefault((mdl, arm), set()).add(blk["prompt_sha"])
    mixed = {k: v for k, v in shas.items() if len(v) > 1}
    print("### workload check\n")
    if mixed:
        print(f"**⚠ {len(mixed)} (model, arm) pairs span MORE THAN ONE prompt_sha** — "
              "the prompt changed mid-campaign and those cells are not comparable:\n")
        for (mdl, arm), v in sorted(mixed.items()):
            print(f"- `{mdl}` / `{arm}`: {sorted(v)}")
    else:
        print(f"✅ every (model, arm) pair has a single `prompt_sha` "
              f"across all geometries and think modes — one workload per cell.")
    hosts = {b.get("host") for (_, _, _, _, _, b) in rows}
    vers = {b.get("server_version") for (_, _, _, _, _, b) in rows}
    print(f"\nhost(s): {sorted(h for h in hosts if h)}  "
          f"build(s): {sorted(v for v in vers if v)}")


if __name__ == "__main__":
    main()
