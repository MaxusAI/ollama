#!/usr/bin/env python3
"""Diff two `CLIP_NODE_STATS` captures — a per-node fingerprint of the vision tower.

Usage: summarize_clip_fingerprint.py [--labels "A,B"] [--tol FLOAT] <a.raw> <b.raw>

Exists to answer "what changed underneath?" after a throughput or score A/B says
something moved. `summarize_tps.py` and the score diff report THAT an arm moved;
this reports WHERE. Compat 801 meters every node of the vision graph, giving a
STRUCTURAL fingerprint (name, op, type) and a NUMERICAL one (max_abs, fp16
headroom `hr`, counts near the fp16 cliff, inf/nan). Read the diff as:

  op or type changed   -> a payload or patch change, not the GPU library
  max_abs drifted      -> kernel numerics changed under an unchanged graph
  n_inf / n_nan appear -> breakage; stop and bisect, do not tune tolerances

Capture with `OLLAMA_CLIP_NODE_STATS='*'` set on the server, then read the
container log. Both arms must run the same model and the same image: this
compares graphs, and a different image is a different graph.

NOT rocBLAS kernel logging. For q4_K_M weights ggml-hip runs its own MMQ kernels
and does not route most matmuls through rocBLAS, so a `ROCBLAS_LAYER` trace comes
back empty — measured on both ROCm 7.2.4 and 10.0.0, not assumed.

COMPARISON IS POSITIONAL, not by name. Node names repeat across a graph (`layer_out-26`
appears four times in a gemma4 capture), so keying a dict by name keeps only the last
occurrence and silently hides any difference in the earlier ones. The first version of
this tool did exactly that and reported 1206 nodes where the capture had 1409.
"""
import argparse, re, statistics as st, sys

FIELD = re.compile(r'(\w+)=([^\s]+)')
STRUCT = ("op", "type")


def parse(path):
    """Ordered list of node records. Order IS the key — see the module docstring."""
    rows = []
    with open(path) as fh:
        for line in fh:
            if "CLIP_NODE_STATS" not in line:
                continue
            d = dict(FIELD.findall(line))
            if "name" in d:
                rows.append(d)
    return rows


def compare(a, b, tol=0.0):
    """(structural, numerical, badness) — positional, with a length guard."""
    n = min(len(a), len(b))
    structural, numerical, bad = [], [], []
    for i in range(n):
        x, y = a[i], b[i]
        if tuple(x.get(f) for f in STRUCT) != tuple(y.get(f) for f in STRUCT):
            structural.append((i, x, y))
        try:
            mx, my = float(x["max_abs"]), float(y["max_abs"])
        except (KeyError, ValueError):
            continue
        denom = max(abs(mx), 1e-9)
        rel = abs(my - mx) / denom
        if rel > tol:
            numerical.append((i, x["name"], mx, my, rel))
    for label, rows in (("a", a), ("b", b)):
        for i, r in enumerate(rows):
            if r.get("n_inf", "0") != "0" or r.get("n_nan", "0") != "0":
                bad.append((label, i, r["name"], r.get("n_inf"), r.get("n_nan")))
    return structural, numerical, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--labels", default="")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="relative max_abs tolerance; 0 means report any difference")
    args = ap.parse_args()
    la, lb = (args.labels.split(",", 1) + [args.a, args.b])[:2] if args.labels else (args.a, args.b)

    a, b = parse(args.a), parse(args.b)
    print(f"| arm | nodes metered |\n|---|---|")
    print(f"| {la} | {len(a)} |\n| {lb} | {len(b)} |\n")
    if len(a) != len(b):
        print(f"⚠ **Node counts differ ({len(a)} vs {len(b)})** — the graphs are not the "
              f"same shape, so a positional diff below compares the first {min(len(a), len(b))} "
              f"only. Check the model and image match before reading anything else.\n")

    structural, numerical, bad = compare(a, b, args.tol)
    print(f"| check | result |\n|---|---|")
    print(f"| structural (op/type) | {'**' + str(len(structural)) + ' differ**' if structural else 'identical'} |")
    print(f"| numerical (max_abs) | {'**' + str(len(numerical)) + ' differ**' if numerical else 'identical'} |")
    print(f"| inf / nan | {'**' + str(len(bad)) + ' nodes**' if bad else 'none'} |")

    if structural:
        print(f"\n**Structural — a payload or patch change, not the GPU library:**\n")
        print("| # | node | " + f"{la} op/type | {lb} op/type |")
        print("|---|---|---|---|")
        for i, x, y in structural[:20]:
            print(f"| {i} | `{x['name']}` | {x.get('op')}/{x.get('type')} | {y.get('op')}/{y.get('type')} |")
    if numerical:
        rels = [t[4] for t in numerical]
        print(f"\n**Numerical — kernel arithmetic changed under an unchanged graph.** "
              f"median relative drift {st.median(rels) * 100:.4f}%, max {max(rels) * 100:.4f}%\n")
        print("| # | node | " + f"{la} max_abs | {lb} max_abs | rel |")
        print("|---|---|---|---|---|")
        for i, nm, mx, my, rel in sorted(numerical, key=lambda t: -t[4])[:20]:
            print(f"| {i} | `{nm}` | {mx:.1f} | {my:.1f} | {rel * 100:+.4f}% |")
    if bad:
        print(f"\n**inf / nan present — this is breakage, not drift:**\n")
        for label, i, nm, ninf, nnan in bad[:20]:
            print(f"- {label} node {i} `{nm}`: n_inf={ninf} n_nan={nnan}")

    if not (structural or numerical or bad):
        print(f"\n**The vision graph is identical between these arms** — same ops, same "
              f"dtypes, same intermediate magnitudes on every one of {len(a)} metered nodes. "
              f"Any scored difference between them originates downstream of the vision "
              f"tower, not inside it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
