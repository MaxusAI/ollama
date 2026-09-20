#!/usr/bin/env python3
"""Render two `run_engine_compare.sh` campaigns as per-test throughput tables.

Usage: summarize_tps.py [--a TAGPREFIX] [--b TAGPREFIX] [--labels "A,B"] [--dir D]
e.g.   summarize_tps.py --a gate4_0342 --b rocm10 --labels "ROCm 7.2.4,ROCm 10.0.0"

Exists because SPEC `vision-harness-reuse.md` H7 requires tables to come from a
generator: a throughput number hand-typed into a document asserts something nothing
checks. H13: the footer is built from the `host` / `server_version` of every block
rendered, and a mix renders the MIXED banner ADR 0012 convention 10 makes
non-publishable.

`prefill_tps` is NOT a clean compute rate. On a KV-cache hit the server still reports
the full `prompt_eval_count` but a collapsed `prompt_eval_duration`, so the ratio
jumps 5-30x while the encoder does almost nothing. Within one campaign the metric is
therefore bimodal, and a median over all blocks reports the arm's cache-hit rate as
if it were speed. Each block is classified `cold` (encode actually ran) or `cache`
(> 2.5x the model's slowest block in that arm -- the two modes are separated by a
wide empty gap, not a close call), and the summary reports the two populations
separately. A block whose class DIFFERS between arms is excluded from the paired
delta and listed: its ratio compares an encode against a cache lookup.

`gen_tps` has no such failure mode and is reported over all warm blocks.

The first block after a container restart is dropped: `cold_start` marks it, and it
carries model-load and clock-ramp cost that no later block pays.
"""
import argparse, glob, json, os, re, statistics as st, sys

NOT_RECORDED = "pre-H11 run (not recorded)"
CACHE_FACTOR = 2.5


def load_arm(d, prefix):
    """{model_slug: {block: record}} for one tag prefix."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, f"scores_{prefix}_1_*_thinkfalse.json"))):
        m = re.match(rf"scores_{re.escape(prefix)}_1_(.+)_thinkfalse\.json$", os.path.basename(p))
        if m:
            with open(p) as f:
                out[m.group(1)] = json.load(f)
    return out


def classify(blocks):
    """cold vs cache, per block. Threshold is relative to the arm's own slowest block."""
    tps = [v["prefill_tps"] for v in blocks.values() if v.get("prefill_tps")]
    if not tps:
        return {}
    lo = min(tps)
    return {k: ("cache" if v["prefill_tps"] > CACHE_FACTOR * lo else "cold")
            for k, v in blocks.items() if v.get("prefill_tps")}


def pct(new, old):
    return None if not old else (new / old - 1.0) * 100.0


def fmt(x, w=1):
    return "—" if x is None else f"{x:,.{w}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="gate4_0342")
    ap.add_argument("--b", default="rocm10")
    ap.add_argument("--labels", default="")
    ap.add_argument("--dir", default=".")
    a = ap.parse_args()

    la, lb = (a.labels.split(",", 1) + [a.a, a.b])[:2] if a.labels else (a.a, a.b)
    A, B = load_arm(a.dir, a.a), load_arm(a.dir, a.b)
    models = [m for m in sorted(set(A) & set(B))]
    if not models:
        sys.exit(f"no models shared between {a.a!r} and {a.b!r} in {a.dir!r}")

    hosts, builds, summary, excluded = set(), set(), [], []

    for m in models:
        ba, bb = A[m], B[m]
        ca, cb = classify(ba), classify(bb)
        keys = [k for k in sorted(set(ba) & set(bb))
                if ba[k].get("cold_start") == "warm" and bb[k].get("cold_start") == "warm"]
        dropped = len(set(ba) & set(bb)) - len(keys)

        print(f"\n### {m}\n")
        print("| test | class | "
              f"gen {la} | gen {lb} | Δ gen | "
              f"prefill {la} | prefill {lb} | Δ prefill | prompt tok |")
        print("|---|---|---|---|---|---|---|---|---|")
        for k in keys:
            ra, rb = ba[k], bb[k]
            hosts.add(ra.get("host") or NOT_RECORDED); hosts.add(rb.get("host") or NOT_RECORDED)
            builds.add(ra.get("server_version") or NOT_RECORDED)
            builds.add(rb.get("server_version") or NOT_RECORDED)
            same = ca.get(k) == cb.get(k)
            cls = ca.get(k, "?") if same else f"**{ca.get(k)}→{cb.get(k)}**"
            if not same:
                excluded.append((m, k, ca.get(k), cb.get(k)))
            dg = pct(rb.get("gen_tps"), ra.get("gen_tps"))
            dp = pct(rb.get("prefill_tps"), ra.get("prefill_tps")) if same else None
            print(f"| {k} | {cls} | {fmt(ra.get('gen_tps'),2)} | {fmt(rb.get('gen_tps'),2)} | "
                  f"{'—' if dg is None else f'{dg:+.1f}%'} | "
                  f"{fmt(ra.get('prefill_tps'))} | {fmt(rb.get('prefill_tps'))} | "
                  f"{'n/c' if dp is None else f'{dp:+.1f}%'} | {ra.get('prompt_eval_count','—')} |")
        if dropped:
            print(f"\n{dropped} block(s) dropped as `cold_start` (first request after restart).")

        row = {"model": m}
        g = [bb[k]["gen_tps"] / ba[k]["gen_tps"] for k in keys
             if ba[k].get("gen_tps") and bb[k].get("gen_tps")]
        row["gen"] = (st.median([ba[k]["gen_tps"] for k in keys if ba[k].get("gen_tps")]),
                      st.median([bb[k]["gen_tps"] for k in keys if bb[k].get("gen_tps")]),
                      (st.median(g) - 1) * 100 if g else None, len(g))
        for want in ("cold", "cache"):
            sel = [k for k in keys if ca.get(k) == want == cb.get(k)]
            r = [bb[k]["prefill_tps"] / ba[k]["prefill_tps"] for k in sel]
            row[want] = (st.median([ba[k]["prefill_tps"] for k in sel]) if sel else None,
                         st.median([bb[k]["prefill_tps"] for k in sel]) if sel else None,
                         (st.median(r) - 1) * 100 if r else None, len(sel))
        summary.append(row)

    print("\n### Summary — median of per-test paired ratios\n")
    print(f"| model | gen {la} | gen {lb} | Δ gen | n | "
          f"prefill {la} (cold) | prefill {lb} (cold) | Δ prefill cold | n | Δ prefill cache | n |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        g, c, h = r["gen"], r["cold"], r["cache"]
        print(f"| {r['model']} | {fmt(g[0],2)} | {fmt(g[1],2)} | "
              f"{'—' if g[2] is None else f'**{g[2]:+.1f}%**'} | {g[3]} | "
              f"{fmt(c[0])} | {fmt(c[1])} | "
              f"{'—' if c[2] is None else f'**{c[2]:+.1f}%**'} | {c[3]} | "
              f"{'—' if h[2] is None else f'{h[2]:+.1f}%'} | {h[3]} |")

    if excluded:
        print("\n**Excluded from the paired prefill delta — cache class differs between arms:**\n")
        for m, k, x, y in excluded:
            print(f"- `{m}` / `{k}`: {x} → {y}")
    else:
        print("\nCache class agreed on every block in every model, "
              "so no block was excluded from the paired prefill delta.")

    if len(hosts) > 1 or len(builds) > 1:
        print(f"\n⚠ **MIXED — rows are not one campaign** "
              f"(hosts: {sorted(hosts)}; builds: {sorted(builds)})"
              if len(hosts) > 1 else f"\nhost: {sorted(hosts)[0]} · builds: {sorted(builds)}")
    else:
        print(f"\nhost: {hosts.pop()} · build: {builds.pop()}")


if __name__ == "__main__":
    main()
