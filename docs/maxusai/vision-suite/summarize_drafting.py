#!/usr/bin/env python3
"""How much a build actually drafted, per model, from one or more runner logs.

    summarize_drafting.py <runner.log> [<runner.log> ...]

Speculative decoding is not a setting you can read off a build: whether a request drafts depends on the request (a
grammar disables it on some builds) and how deep it drafts is decided at runtime by a timing-driven controller that
starts at depth 0 after every load. So "this build drafts" is a measurement, and this is where it comes from.

Columns. `completions` counts requests that could have drafted; `with stats` counts those that logged a draft-stats
line. The two together are the point: 0 of 28 means the build refused to draft, while 28 of 28 with a low
`tokens/round` means it drafted and the drafts rarely paid. `tokens/round` is what each target forward commits — the
round's own token plus the drafts it kept — so 1.00 is plain decode. `max depth` is the deepest draft chain seen.

An exploratory table by ADR 0012 rule 7: it reports the runner's own counters, not suite scores.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runnerlog  # noqa: E402  (ADR 0028 rule 3: one runner-log parser)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        per_model = {}
        for r in runnerlog.iter_requests(path):
            agg = per_model.setdefault(r.model, {"stats": 0, "rounds": 0, "drafted": 0, "accepted": 0, "depth": 0})
            if r.rounds is None:
                continue
            agg["stats"] += 1
            agg["rounds"] += r.rounds
            agg["drafted"] += r.drafted
            agg["accepted"] += r.accepted
            agg["depth"] = max(agg["depth"], r.max_draft)
        completions = runnerlog.completions(path)
        print(f"### `{os.path.basename(path)}`\n")
        print("| model | completions | with stats | decode rounds | drafted/round | acceptance | tokens/round | max depth |")
        print("|---|---|---|---|---|---|---|---|")
        for model in completions or per_model:
            a = per_model.get(model, {"stats": 0, "rounds": 0, "drafted": 0, "accepted": 0, "depth": 0})
            n, rounds = a["stats"], a["rounds"]
            if not n or not rounds:
                print(f"| {model} | {completions.get(model, 0)} | 0 | — | — | — | — | — |")
                continue
            print(f"| {model} | {completions.get(model, 0)} | {n} | {rounds} | {a['drafted'] / rounds:.3f} | "
                  f"{a['accepted'] / a['drafted']:.2f} | {(rounds + a['accepted']) / rounds:.3f} | {a['depth']} |")
        print()


if __name__ == "__main__":
    main()
