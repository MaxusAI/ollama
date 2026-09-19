#!/usr/bin/env python3
"""Render `extbench.py` runs as one markdown table, with an H13 provenance footer.

Usage: summarize_extbench.py [--dir D] [--paired] <bench> <tag> [tag ...]
e.g.   summarize_extbench.py ocrbench ocr1np_q4_k_m ocr1np_q8_0 ocr1np_bf16

Exists because SPEC `vision-harness-reuse.md` H7 requires tables to be emitted by a
generator and pasted verbatim: an external-benchmark score that is hand-typed into a
document asserts a number nothing checks. `summarize_engine_compare.py` renders the
in-house suite; nothing rendered `ext_*.json` until this.

H5: `load` and `fmt_bool` are imported from `summarize_engine_compare.py`, never
redefined.

H13: the footer is built from the `host` / `server_version` fields of every file
rendered. Files written before extbench recorded them show as **pre-H11 run (not
recorded)** rather than silently inheriting a sibling's provenance, and a mix of
hosts, builds or recording states renders the MIXED banner that ADR 0012
convention 10 makes non-publishable.

With `--paired`, also runs McNemar's exact test between every pair of arms over the
items both scored. Aggregate accuracies cannot distinguish "these backends differ"
from "they disagree on a handful of coin-flips"; on a shared row set the discordant
pairs can.
"""
import json
import math
import os
import sys

from summarize_engine_compare import fmt_bool, load  # noqa: F401  (H5)

DIR = os.path.dirname(os.path.abspath(__file__))

NOT_RECORDED = "pre-H11 run (not recorded)"


def path_for(bench, tag, rundir=DIR):
    return os.path.join(rundir, f"ext_{tag}_{bench}.json")


def provenance(summary):
    """(hosts, builds) as displayable strings. Absent fields are NOT guessed."""
    def one(key):
        v = summary.get(key)
        if not v:
            return NOT_RECORDED
        return ", ".join(v) if isinstance(v, list) else str(v)
    return one("host"), one("server_version")


def mcnemar_exact(b, c):
    """Two-sided exact binomial p for b discordant one way, c the other.

    Exact rather than chi-square because the discordant counts here are single
    digits. The uncorrected chi-square is anticonservative there -- 3-vs-1 reads
    0.317 against the true 0.625 -- and while the continuity correction recovers
    most of it (0.617), the exact test needs no such caveat and costs nothing at
    these counts.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    argv = list(sys.argv[1:])
    paired = "--paired" in argv
    argv = [a for a in argv if a != "--paired"]
    rundir = DIR
    if "--dir" in argv:
        k = argv.index("--dir")
        rundir = argv[k + 1]
        del argv[k:k + 2]
    if len(argv) < 2:
        sys.exit(__doc__)
    bench, tags = argv[0], argv[1:]

    rows, hosts, builds, per_item = [], set(), set(), {}
    for tag in tags:
        d = load(path_for(bench, tag, rundir))
        if d is None:
            print(f"<!-- missing: {path_for(bench, tag, rundir)} -->")
            continue
        s = d["summary"]
        h, b = provenance(s)
        hosts.add(h)
        builds.add(b)
        rows.append(s)
        per_item[tag] = {r["i"]: bool(r.get("ok")) for r in d["records"] if "error" not in r}

    if not rows:
        sys.exit("no score files loaded")

    acc_key = "acc@0.5" if "acc@0.5" in rows[0] else "accuracy"
    print(f"| model | scored | errors | empty | correct | {acc_key} | think | endpoint |")
    print("|---|---|---|---|---|---|---|---|")
    for s in rows:
        print(f"| `{s['model']}` | {s['scored']} | {s['errors']} | {s['empty_responses']} | "
              f"{s['correct']} | **{s[acc_key]}** | {s['think_env']} | {s['endpoint']} |")

    r0 = rows[0]
    print(f"\n{r0['benchmark']} — `{r0['dataset']}` [{r0['split']}], "
          f"rows {r0['offset']}..{r0['offset'] + r0['requested']}.")

    if len(hosts) > 1 or len(builds) > 1:
        print("\n⚠ **MIXED — rows are not one campaign** "
              f"(hosts: {sorted(hosts)}; builds: {sorted(builds)})")
    else:
        print(f"\nhost: {hosts.pop()} · build: {builds.pop()}")

    if paired and len(tags) > 1:
        print(f"\n| pair | both ✓ | both ✗ | A only | B only | McNemar exact p |")
        print("|---|---|---|---|---|---|")
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                a, b = tags[i], tags[j]
                shared = sorted(set(per_item[a]) & set(per_item[b]))
                both = sum(1 for k in shared if per_item[a][k] and per_item[b][k])
                neither = sum(1 for k in shared if not per_item[a][k] and not per_item[b][k])
                aonly = sum(1 for k in shared if per_item[a][k] and not per_item[b][k])
                bonly = sum(1 for k in shared if not per_item[a][k] and per_item[b][k])
                p = mcnemar_exact(aonly, bonly)
                print(f"| {a} vs {b} | {both} | {neither} | {aonly} | {bonly} | {p:.3f} |")


if __name__ == "__main__":
    main()
