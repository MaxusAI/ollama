#!/usr/bin/env python3
"""Render an OCRBench ladder from `extbench.py` score files (ADR 0012: tables come from generators).

    python3 ocrbench_table.py "nvfp4=ocr31b_nvfp4,ocr31b_nvfp4_r2" "q4_K_M=ocr31b_q4km"
    python3 ocrbench_table.py --dir . --engine mlx-cuda "bf16=ocr31b_bf16"

Each argument is `label=tag[,tag...]`; several tags are repeats of one arm. Reads
`ext_<tag>_ocrbench.json` and writes three tables: the arms, the repeat spread, and
every pair compared on the items both arms answered.

Why paired: a 200-item slice carries a binomial standard error near 2.4 points at
p = 0.87, so two arms 3 items apart are indistinguishable from their accuracies alone.
The arms answer the *same* items, so the discordant pairs (b, c) carry the signal and
an exact McNemar test resolves differences the marginals cannot. A difference that does
not clear it is not a difference; report it as unresolved rather than as a ranking.
"""
import argparse
import json
import math
import os
import sys
from itertools import combinations


def load(directory, tag):
    with open(os.path.join(directory, f"ext_{tag}_ocrbench.json")) as f:
        d = json.load(f)
    rec = [r for r in d["records"] if "error" not in r]
    return {
        "summary": d["summary"],
        "ok": {r["i"]: bool(r["ok"]) for r in rec},
        "secs": [r.get("secs") or 0 for r in rec],
        "prompt_eval": [r.get("prompt_eval_count") or 0 for r in rec],
    }


def stderr(correct, n):
    """Binomial standard error of the slice accuracy."""
    if not n:
        return 0.0
    p = correct / n
    return math.sqrt(p * (1 - p) / n)


def mcnemar_p(b, c):
    """Exact two-sided McNemar: the discordant pairs under a fair coin."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def median(xs):
    return sorted(xs)[len(xs) // 2] if xs else 0.0


def arm_table(arms, engine):
    rows = ["| arm | engine | model | correct / scored | accuracy | ±1 s.e. | mean s/item | median | prompt_eval |",
            "|---|---|---|---|---|---|---|---|---|"]
    for label, runs in arms.items():
        for i, (tag, r) in enumerate(runs):
            s = r["summary"]
            n, c = s["scored"], s["correct"]
            name = label if len(runs) == 1 else f"{label} (run {i + 1})"
            rows.append(f"| {name} | {engine} | `{s['model']}` | {c} / {n} | **{c / n:.3f}** | "
                        f"{stderr(c, n):.3f} | {mean(r['secs']):.1f} | {median(r['secs']):.1f} | "
                        f"{mean(r['prompt_eval']):.0f} |")
    return "\n".join(rows)


def repeat_table(arms):
    """Repeats of one arm: the spread, and how many items changed verdict between runs."""
    multi = {l: r for l, r in arms.items() if len(r) > 1}
    if not multi:
        return ""
    rows = ["| arm | runs | accuracies | items that flipped |", "|---|---|---|---|"]
    for label, runs in multi.items():
        accs = [f"{r['summary']['correct'] / r['summary']['scored']:.3f}" for _, r in runs]
        base = runs[0][1]["ok"]
        flips = sum(sum(1 for i, v in r["ok"].items() if i in base and v != base[i])
                    for _, r in runs[1:])
        rows.append(f"| {label} | {len(runs)} | {', '.join(accs)} | {flips} |")
    return "\n".join(rows)


def paired_table(arms):
    """Every pair on the items both answered: b, c and the exact McNemar p."""
    first = {l: r[0][1] for l, r in arms.items()}
    rows = ["| A | B | A | B | b (A only) | c (B only) | p | resolved |",
            "|---|---|---|---|---|---|---|---|"]
    for (la, ra), (lb, rb) in combinations(first.items(), 2):
        shared = sorted(set(ra["ok"]) & set(rb["ok"]))
        if not shared:
            continue
        b = sum(1 for i in shared if ra["ok"][i] and not rb["ok"][i])
        c = sum(1 for i in shared if rb["ok"][i] and not ra["ok"][i])
        p = mcnemar_p(b, c)
        rows.append(f"| {la} | {lb} | {sum(ra['ok'][i] for i in shared) / len(shared):.3f} | "
                    f"{sum(rb['ok'][i] for i in shared) / len(shared):.3f} | {b} | {c} | "
                    f"{p:.3f} | {'**yes**' if p < 0.05 else 'no'} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("arms", nargs="+", help="label=tag[,tag...]")
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--engine", default="mlx-cuda",
                    help="engine label for the table; mlx-cuda and mlx-metal numbers never share one")
    a = ap.parse_args(argv)

    arms, missing = {}, []
    for spec in a.arms:
        label, _, tags = spec.partition("=")
        runs = []
        for tag in tags.split(","):
            try:
                runs.append((tag, load(a.dir, tag)))
            except FileNotFoundError:
                missing.append(tag)
        if runs:
            arms[label] = runs
    if not arms:
        sys.exit(f"no score files found (looked for {', '.join(missing)} in {a.dir})")

    s = arms[next(iter(arms))][0][1]["summary"]
    print(arm_table(arms, a.engine))
    print(f"\nOCRBench `{s['dataset']}` [{s['split']}] rows {s['offset']}–{s['offset'] + s['requested']}, "
          f"think {s['think_env']}, endpoint {s['endpoint']}, contains-match scoring (lmms-eval semantics).")
    rep = repeat_table(arms)
    if rep:
        print("\n**Repeats**\n")
        print(rep)
    if len(arms) > 1:
        print("\n**Paired on the same items** (first run of each arm)\n")
        print(paired_table(arms))
    if missing:
        print(f"\nMissing score files: {', '.join('`ext_%s_ocrbench.json`' % m for m in missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
