#!/usr/bin/env python3
"""Which tests answered differently between two campaigns, by answer length.

    summarize_output_lengths.py [--dir RUNDIR] <prefix-a> <prefix-b> <model> [<model> ...] [--think false|on]

Scores carry what a scorer measured, not the text, so two runs can agree on every scored cell while the model said
something different. This compares each test's `eval_count` and `answer_chars` between two campaigns and reports the
tests where they differ.

Read it as a floor, in one direction only: equal lengths do NOT prove equal text, and different lengths DO prove
different text. That is what makes it useful as a noise floor — two runs of ONE build differ on a few tests, and a
cross-build difference inside that band is not evidence of anything. On the v0.34.0 fold, two runs of main differed
on 2 of 27 gemma4 tests and 7 of 27 qwen3.6 tests, which is the same size as main against the fold.

Tags are built by `summarize_engine_compare.resolve_tag`, so a prefixed campaign resolves exactly as it does for
T1/T2 and the think-suffix fallback stays in one place (ADR 0028 rule 3).

An exploratory table by ADR 0012 rule 7.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_engine_compare import resolve_tag  # noqa: E402  (ADR 0028 rule 3: one tag builder)

DIR = os.path.dirname(os.path.abspath(__file__))


def blocks(rundir, prefix, model, think):
    path = os.path.join(rundir, f"scores_{resolve_tag(rundir, model, think, prefix)}.json")
    try:
        with open(path) as fh:
            return json.load(fh) or {}, os.path.basename(path)
    except OSError:
        return None, os.path.basename(path)


def main():
    argv, think, rundir = sys.argv[1:], "false", DIR
    if "--dir" in argv:
        i = argv.index("--dir")
        rundir = argv[i + 1]
        del argv[i:i + 2]
    if "--think" in argv:
        i = argv.index("--think")
        think = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) < 3:
        sys.exit(__doc__)
    a, b, models = argv[0], argv[1], argv[2:]
    print(f"| model | tests in both | same length | differ | tests that differ |")
    print("|---|---|---|---|---|")
    for model in models:
        da, fa = blocks(rundir, a, model, think)
        db, fb = blocks(rundir, b, model, think)
        if da is None or db is None:
            print(f"| {model} | missing `{fa if da is None else fb}` | | | |")
            continue
        common = [t for t in da if isinstance(da[t], dict) and isinstance(db.get(t), dict) and "eval_count" in da[t]]
        differ = [t for t in common
                  if (da[t].get("eval_count"), da[t].get("answer_chars")) != (db[t].get("eval_count"), db[t].get("answer_chars"))]
        print(f"| {model} | {len(common)} | {len(common) - len(differ)} | {len(differ)} | {', '.join(differ) or '—'} |")


if __name__ == "__main__":
    main()
