#!/usr/bin/env python3
"""Render `extbench.py` runs as one markdown table, with an H13 provenance footer.

Usage: summarize_extbench.py [--dir D] [--paired] [--repeats] [--categories]
                             [--timing] <bench> <tag|label=tag[,tag] ...>
e.g.   summarize_extbench.py ocrbench ocr1np_q4_k_m ocr1np_q8_0 ocr1np_bf16
       summarize_extbench.py --repeats --categories ocrbench "q4=ocr_q4,ocr_q4_r2"

A bare argument is one arm. `label=tag,tag` is one arm measured more than once:
the extra tags are repeats, and `--repeats` reports the spread and how many items
changed verdict between them. The default table is unchanged by any of these flags,
so a document that pasted it (H7) stays valid.

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

`--repeats` is the control that makes the paired test readable: an arm re-run against
itself should move less than the arms move against each other, and on a 200-item
OCRBench slice both engines have come back with zero items changed. `--categories`
splits each arm by the benchmark's own question type, read from the row slice
`extbench.py` cached, which is what H19 asks a slice result to state. `--timing` adds
seconds per item, mean prompt_eval, and the binomial standard error of the accuracy —
at 200 items that is about 2.4 points, which is why the paired columns exist.
"""
import json
import math
import os
import sys

from summarize_engine_compare import fmt_bool, load  # noqa: F401  (H5)

DIR = os.path.dirname(os.path.abspath(__file__))

NOT_RECORDED = "pre-H11 run (not recorded)"


def parse_arm(spec):
    """`label=tag,tag` -> (label, [tags]); a bare `tag` -> (tag, [tag])."""
    if "=" in spec:
        label, _, tags = spec.partition("=")
        return label, [t for t in tags.split(",") if t]
    return spec, [spec]


def stderr(correct, n):
    """Binomial standard error of a slice accuracy."""
    if not n:
        return 0.0
    p = correct / n
    return math.sqrt(p * (1 - p) / n)


def question_types(rundir, bench, offset, limit):
    """{row index: question_type} from extbench's cached slice; {} when absent."""
    path = os.path.join(rundir, "extimgs", bench, f"rows_{offset}_{limit}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {offset + i: r.get("question_type") or "unlabelled" for i, r in enumerate(rows)}


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
    flags = {f: f"--{f}" in argv for f in ("paired", "repeats", "categories", "timing")}
    paired = flags["paired"]
    argv = [a for a in argv if not a.startswith("--") or a == "--dir"]
    rundir = DIR
    if "--dir" in argv:
        k = argv.index("--dir")
        rundir = argv[k + 1]
        del argv[k:k + 2]
    if len(argv) < 2:
        sys.exit(__doc__)
    bench, arms = argv[0], [parse_arm(a) for a in argv[1:]]
    tags = [a[1][0] for a in arms]            # the first run of each arm

    rows, hosts, builds, per_item, runs = [], set(), set(), {}, {}
    for label, arm_tags in arms:
        for tag in arm_tags:
            d = load(path_for(bench, tag, rundir))
            if d is None:
                print(f"<!-- missing: {path_for(bench, tag, rundir)} -->")
                continue
            s = d["summary"]
            h, b = provenance(s)
            hosts.add(h)
            builds.add(b)
            ok = {r["i"]: bool(r.get("ok")) for r in d["records"] if "error" not in r}
            recs = [r for r in d["records"] if "error" not in r]
            runs.setdefault(label, []).append((tag, s, ok, recs))
            if tag == arm_tags[0]:
                rows.append(s)
                per_item[tag] = ok

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

    if flags["repeats"] and any(len(v) > 1 for v in runs.values()):
        print("\n| arm | runs | accuracies | items that changed verdict |")
        print("|---|---|---|---|")
        for label, got in runs.items():
            if len(got) < 2:
                continue
            accs = ", ".join(f"{g[1]['correct'] / g[1]['scored']:.3f}" for g in got)
            base = got[0][2]
            flips = sum(sum(1 for i, v in g[2].items() if i in base and v != base[i])
                        for g in got[1:])
            print(f"| {label} | {len(got)} | {accs} | {flips} |")

    if flags["timing"]:
        print("\n| arm | accuracy | ±1 s.e. | mean s/item | median | mean prompt_eval |")
        print("|---|---|---|---|---|---|")
        for label, got in runs.items():
            _, s0, _, recs = got[0]
            secs = [r.get("secs") or 0 for r in recs]
            pe = [r.get("prompt_eval_count") or 0 for r in recs]
            med = sorted(secs)[len(secs) // 2] if secs else 0
            print(f"| {label} | {s0['correct'] / s0['scored']:.3f} | "
                  f"{stderr(s0['correct'], s0['scored']):.3f} | "
                  f"{(sum(secs) / len(secs) if secs else 0):.1f} | {med:.1f} | "
                  f"{(sum(pe) / len(pe) if pe else 0):.0f} |")

    if flags["categories"]:
        types = question_types(rundir, bench, r0["offset"], r0["requested"])
        if not types:
            print(f"\n<!-- no cached row slice in extimgs/{bench}; nothing to split by -->")
        else:
            labels = list(runs)
            print("\n| question type | n | " + " | ".join(labels) + " |")
            print("|---|---|" + "---|" * len(labels))
            for name in sorted(set(types.values())):
                idx = [i for i, t in types.items() if t == name]
                cells = []
                for label in labels:
                    ok = runs[label][0][2]
                    got = [ok[i] for i in idx if i in ok]
                    cells.append(f"{sum(got)}/{len(got)}" if got else "—")
                print(f"| {name} | {len(idx)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
