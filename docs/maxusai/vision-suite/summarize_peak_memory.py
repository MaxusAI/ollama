#!/usr/bin/env python3
"""Per-request peak memory, one runner log against another.

    summarize_peak_memory.py <base-runner.log> <other-runner.log> [model ...]

Both logs must come from the SAME request sequence — the same suite in the same order — because this pairs requests
by position. The runner resets MLX's peak at the start of every request and logs it at teardown, so each number is
that request's own peak, not a high-water mark for the run.

Why per request and not the maximum: a build can raise the peak of every small request and still show the same
maximum, because the maximum belongs to the largest request in the suite. That is exactly what drafting did to
qwen3.8 in the v0.34.0 fold — median +2.9 GiB with the maximum moved by 1.7.

When the other log carries draft-stats lines, the split by draft depth is reported too: a difference that appears
only where drafting was deep is evidence about drafting, and one that is flat across depths is not.

An exploratory table by ADR 0012 rule 7.
"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runnerlog  # noqa: E402  (ADR 0028 rule 3: one runner-log parser)


def by_model(path):
    out = {}
    for r in runnerlog.iter_requests(path):
        out.setdefault(r.model, []).append(r)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    base, other = by_model(sys.argv[1]), by_model(sys.argv[2])
    models = sys.argv[3:] or [m for m in base if m in other]
    print("| model | paired requests | max peak | median diff | largest diff | requests > +5 GiB |")
    print("|---|---|---|---|---|---|")
    notes = []
    for m in models:
        a, b = base.get(m, []), other.get(m, [])
        n = min(len(a), len(b))
        if not n:
            print(f"| {m} | 0 | — | — | — | — |")
            continue
        diffs = [runnerlog.gib(b[i].peak) - runnerlog.gib(a[i].peak) for i in range(n)]
        print(f"| {m} | {n} | {runnerlog.gib(max(x.peak for x in a)):.2f} → "
              f"{runnerlog.gib(max(x.peak for x in b)):.2f} GiB | {statistics.median(diffs):+.2f} GiB | "
              f"{max(diffs, key=abs):+.2f} GiB | {sum(1 for d in diffs if d > 5)} |")
        deep = [diffs[i] for i in range(n) if b[i].avg_draft is not None and b[i].avg_draft >= 4]
        shallow = [diffs[i] for i in range(n) if b[i].avg_draft is not None and b[i].avg_draft < 1]
        if deep or shallow:
            parts = []
            if shallow:
                parts.append(f"avg_draft < 1: median {statistics.median(shallow):+.2f} GiB (n={len(shallow)})")
            if deep:
                parts.append(f"avg_draft ≥ 4: median {statistics.median(deep):+.2f} GiB (n={len(deep)})")
            notes.append(f"- **{m}**, split by the other log's draft depth: " + "; ".join(parts))
    if notes:
        print()
        print("\n".join(notes))


if __name__ == "__main__":
    main()
