#!/usr/bin/env python3
"""Render the composable bbox matrix as MARGINAL EFFECTS, not 1,120 rows.

    python3 summarize_matrix.py [--dir D] [--metric hits_declared] <tag-prefix>

The matrix is 8 arms (pin × anchor × coords) × think × geometry × model. Printed
flat that is 1,120 rows nobody reads. What the design is FOR is the effect of each
factor, so this pools the other axes and reports the marginal. Template T5
(ADR 0012).

**The pooling is the dangerous part and it is done explicitly.** A marginal that
averages over a factor which INTERACTS with the one under test is a lie with a
number attached. So every marginal is reported per model and per think mode, and
only geometry is pooled — because geometry is the axis the contract SPEC already
measured as invariant under the pin (111 of 112 cells). If that stops being true
the per-geometry spread column will show it, and the ⚠ names the worst geometry
so the flag is actionable rather than wallpaper.

**Capped cells are excluded from every mean and counted per level.** A cell
whose eval_count reached req_num_predict measures the harness cap, not the
model — the same rule as summarize_engine_compare.was_capped(). Pooling them in
flattens two different failure modes, "cannot ground" and "cannot stop", into
one number: measured 2026-08-20, qwen3.6 think-on was 60/80 capped and its
pooled marginal was mostly termination failure wearing a grounding score. The
count is per LEVEL, not per row, because termination itself may respond to the
factor (the anchor bounded qwen3.6's multi_3img runaway from >122,880 tokens to
10,910 — the capped column is where that effect would show here).

**A partial campaign renders as INCOMPLETE, with the missing cells named.**
ADR 0012 rule 8: re-render after a campaign fully completes. The expected grid
is the cross-product of every model, mode and geometry seen on disk × 8 arms,
so a sweep that has not reached a geometry yet is visible instead of silently
shrinking n (the "53 of 54" class of error).

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_engine_compare import was_capped  # noqa: E402  (SPEC H5)

FACTORS = ("pin", "anchor", "coords")
ARM_RE = re.compile(r"^bboxm_(pin|free)_(anc|noanc)_(named|pos)$")
ARMS = [f"bboxm_{p}_{a}_{c}"
        for p in ("pin", "free") for a in ("anc", "noanc") for c in ("named", "pos")]
# Δ direction is part of the claim; an unlabeled sign made nemotron's −2.53 on
# coords ("positional better") look like a typo.
DELTA = {"pin": "Δ (on−off)", "anchor": "Δ (on−off)", "coords": "Δ (named−pos)"}


def factors_of(arm):
    m = ARM_RE.match(arm)
    if not m:
        return None
    p, a, c = m.groups()
    return {"pin": p == "pin", "anchor": a == "anc", "coords": c}


def capped(blk):
    """True when generation stopped at the cap rather than finishing.
    Such a cell's score is a harness setting, not a model result.

    Delegates to the ONE definition (SPEC H5): the server's done_reason wins,
    with the arithmetic fallback for pre-2026-08-20 blocks. This module's own
    req_num_predict arithmetic predated done_reason and misread the
    synthetic-length class (a window-bound continuation reports "length"
    below the cap), letting exactly the cells ADR 0012 conv 9 forbids into
    the pooled marginals. Blocks here carry req_num_predict rather than
    num_predict, which the fallback reads — hence the aliasing."""
    if not blk.get("num_predict") and blk.get("req_num_predict"):
        blk = dict(blk, num_predict=blk["req_num_predict"])
    return was_capped(blk)


def load(rundir, prefix):
    """[(model, mode, geometry, arm, factors, block, capped)]"""
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
                out.append((model, mode, geom, arm, fac, blk, capped(blk)))
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
    n_cap = sum(1 for r in rows if r[6])
    n_salv = sum(1 for r in rows if r[5].get("json_salvaged"))
    print(f"\n**{a.metric}** — {len(rows)} arm-results, "
          f"{len(models)} models × {len(modes)} think modes × {len(geoms)} geometries")
    print(f"capped (excluded from every mean): {n_cap}  ·  json salvaged: {n_salv}\n")

    # ---- completeness: a partial campaign must say so (ADR 0012 rule 8) -----
    present = {(r[0], r[1], r[2], r[3]) for r in rows}
    expected = {(m, mo, g, arm) for m in models for mo in modes
                for g in geoms for arm in ARMS}
    missing = expected - present
    if missing:
        by = {}
        for (m, mo, g, arm) in missing:
            by.setdefault(mo, {}).setdefault(g, 0)
            by[mo][g] += 1
        print(f"⚠ **INCOMPLETE** — {len(present)}/{len(expected)} cells. "
              "Do not publish; re-render when the campaign completes. Missing:")
        for mo in sorted(by):
            gs = ", ".join(f"{g}({n})" for g, n in sorted(by[mo].items()))
            print(f"- think={mo}: {gs}")
        print()

    # ---- one window rung PER POOLING GROUP, or its mean is not one result ---
    # The check is per (model, mode) because that is the unit every mean pools
    # within; the two think modes legitimately run different generation
    # allowances (2200 off / 8192 on) and are never averaged together, so a
    # whole-dataset check would cry wolf on every campaign.
    gw = {}
    for r in rows:
        gw.setdefault((r[0], r[1]), set()).add(
            (r[5].get("req_num_ctx"), r[5].get("req_num_predict")))
    spans = {k: v for k, v in gw.items() if len(v) > 1}
    if spans:
        print("⚠ **MIXED WINDOWS inside a pooling group** — these means mix "
              "ladder rungs and are not one result (ADR 0012 rule 8):\n")
        for (mdl, mode), v in sorted(spans.items()):
            print(f"- `{mdl}` think={mode}: {sorted(v)}")
        print()
    wins = {mo: sorted({w for (m, m2), v in gw.items() if m2 == mo for w in v})
            for mo in modes}

    # ---- marginal effect of each factor, per model × mode -------------------
    for factor in FACTORS:
        levels = (["named", "pos"] if factor == "coords" else [True, False])
        lab = {True: "on", False: "off", "named": "named", "pos": "positional"}
        print(f"### factor: {factor}\n")
        print("| model | think | " + " | ".join(f"{lab[l]}" for l in levels)
              + f" | {DELTA[factor]} | capped ({lab[levels[0]]}·{lab[levels[1]]})"
              " | geometry spread |")
        print("|---|---|" + "---|" * (len(levels) + 3))
        for mdl in models:
            for mode in modes:
                cells = [r for r in rows if r[0] == mdl and r[1] == mode]
                means, spreads, caps, worst = [], [], [], []
                for lv in levels:
                    lvc = [r for r in cells if r[4][factor] == lv]
                    caps.append(f"{sum(1 for r in lvc if r[6])}/{len(lvc)}")
                    vs = [val(b, a.metric) for (_, _, _, _, _, b, c) in lvc
                          if not c and val(b, a.metric) is not None]
                    means.append(statistics.mean(vs) if vs else None)
                    # spread ACROSS GEOMETRIES at this level — the pooling check
                    per_g = {}
                    for (_, _, g, _, _, b, c) in lvc:
                        if not c and val(b, a.metric) is not None:
                            per_g.setdefault(g, []).append(val(b, a.metric))
                    gm = {g: statistics.mean(v) for g, v in per_g.items()}
                    if len(gm) > 1:
                        med = statistics.median(gm.values())
                        w = max(gm, key=lambda g: abs(gm[g] - med))
                        spreads.append(max(gm.values()) - min(gm.values()))
                        worst.append(w)
                    else:
                        spreads.append(0.0)
                        worst.append(None)
                if not any(m is not None for m in means):
                    continue
                d = (means[0] - means[1]) if all(m is not None for m in means) else None
                sp = max(spreads) if spreads else 0.0
                # A wide geometry spread means pooling hides something — say WHERE.
                wgeom = worst[spreads.index(sp)] if spreads else None
                warn = f" ⚠ {wgeom}" if sp > 1.0 and wgeom else ""
                fm = lambda v: f"{v:.2f}" if v is not None else "—"
                print(f"| {mdl.replace('_','.')[:22]} | {mode} | "
                      + " | ".join(fm(m) for m in means)
                      + f" | {fm(d)} | {caps[0]} · {caps[1]} | {sp:.2f}{warn} |")
        print()

    # ---- the dialect confound, stated rather than pooled over ---------------
    keys = {}
    for (mdl, _, _, arm, fac, blk, _) in rows:
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

    # ---- the decision view: which config can actually be shipped ------------
    # "Trustable" is strict: every geometry present, no cell capped, and the
    # highest mean with the smallest geometry spread. A model whose every arm
    # hit the cap has no trustable config in that mode — that IS the answer,
    # not a rendering gap.
    print("### trustable configs — best arm per model, uncapped cells only\n")
    print("| model | think | arm | mean | geometry spread | n |")
    print("|---|---|---|---|---|---|")
    for mdl in models:
        for mode in modes:
            cand = []
            for arm in ARMS:
                ac = [r for r in rows if r[0] == mdl and r[1] == mode and r[3] == arm]
                if not ac or any(r[6] for r in ac):
                    continue  # capped anywhere → not trustable in this mode
                vs = [val(b, a.metric) for (_, _, _, _, _, b, _) in ac
                      if val(b, a.metric) is not None]
                if not vs:
                    continue
                per_g = {}
                for (_, _, g, _, _, b, _) in ac:
                    if val(b, a.metric) is not None:
                        per_g.setdefault(g, []).append(val(b, a.metric))
                gm = [statistics.mean(v) for v in per_g.values()]
                sp = max(gm) - min(gm) if len(gm) > 1 else 0.0
                cand.append((statistics.mean(vs), sp, arm, len(vs)))
            if not cand:
                if any(r[0] == mdl and r[1] == mode for r in rows):
                    print(f"| {mdl.replace('_','.')[:22]} | {mode} | — | — | — | "
                          "every arm hit the cap at least once |")
                continue
            cand.sort(key=lambda t: (-t[0], t[1]))
            mean, sp, arm, n = cand[0]
            print(f"| {mdl.replace('_','.')[:22]} | {mode} | `{arm}` "
                  f"| {mean:.2f} | {sp:.2f} | {n} |")
    print()

    # ---- workload identity: one prompt_sha per (arm, model) or the run is mixed
    shas = {}
    for (mdl, _, _, arm, _, blk, _) in rows:
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
    hosts = {b.get("host") for (_, _, _, _, _, b, _) in rows}
    vers = {b.get("server_version") for (_, _, _, _, _, b, _) in rows}
    print(f"\nhost(s): {sorted(h for h in hosts if h)}  "
          f"build(s): {sorted(v for v in vers if v)}  "
          f"window(s) (num_ctx, num_predict): "
          + "; ".join(f"think={mo}: {wins[mo]}" for mo in modes))


if __name__ == "__main__":
    main()
