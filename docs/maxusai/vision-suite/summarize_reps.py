#!/usr/bin/env python3
"""Aggregate repeated runs of one arm into mean and spread.

ADR 0023 asks for n=3 per think-on cell because on-policy sampling is not
deterministic. Three scores files then have to become one row, and doing that
arithmetic by hand into a table is precisely what ADR 0012 rule 8 forbids —
"never transcribe generator output by hand". So this does it.

    python3 summarize_reps.py rocm-n3-qwen38-thinkoff rocm-n3-qwen38-thinkon

Tags are arm names. Files are matched as scores_<tag>.json and
scores_<tag>-rep<N>.json, so an arm measured once and an arm measured three
times are both handled, and the reported n says which.

Spread is max-min, not a standard deviation: with n=3 the range is what a reader
can actually reason about, and it is the number that tells you whether a
difference between two arms survives the noise within one arm.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_engine_compare import was_capped  # noqa: E402  (SPEC H5)

DIR = os.path.dirname(os.path.abspath(__file__))

# (section, key, label, kind). "bool" is scored as a pass count, not a mean.
METRICS = [
    ("scene_single", "bbox_mean_iou", "scene bbox IoU", "float"),
    ("scene_single", "labels_found", "scene labels", "int"),
    ("scene_single", "colors_right", "scene colors", "int"),
    ("scene_single", "serial_found", "scene serial", "bool"),
    ("document_single", "items_found", "doc items", "int"),
    ("document_single", "qty_price_right", "doc qty+price", "int"),
    ("document_single", "total_right", "doc total", "bool"),
    ("document_single", "name_bbox_mean_iou", "doc name_bbox IoU", "float"),
    ("multi_3img", "q1_right", "multi q1", "bool"),
    ("multi_3img", "q2_right", "multi q2", "bool"),
    ("multi_3img", "q4_bbox_hit", "multi q4-bbox", "bool"),
    ("multi_3img", "chart_values_found", "multi chart", "int"),
    ("finetext", "recall_22px", "finetext 22px", "int"),
    ("finetext", "recall_16px", "finetext 16px", "int"),
    ("finetext", "recall_12px", "finetext 12px", "int"),
    ("finetext", "recall_9px", "finetext 9px", "int"),
    ("finetext", "recall_7px", "finetext 7px", "int"),
    # Derived, and the honest way to read the tiers. Every model observed so far
    # emits all 20 codes regardless of what it can resolve (finetext_probe.py
    # says so), so a zeroed small tier means FABRICATED codes, not omitted ones.
    # Per-tier recall alone reads as "lost the 9px tier" when the tiers can swap
    # between runs while the total correct stays put — that swap is which
    # fabrications happened to coincide with ground truth, not a capability
    # change. Measured 2026-08-17: two think-on runs scored 4/4/4/1/0 and
    # 4/4/4/0/1, identical totals, opposite tiers.
    ("finetext", "_correct", "finetext correct /20", "int"),
    # total_found - sum(recall): entries that matched no gold code. That
    # includes malformed and duplicate entries, not only invented ones, so the
    # label says unmatched rather than claiming to know which.
    ("finetext", "_fabricated", "finetext unmatched", "int"),
]

TIERS = (22, 16, 12, 9, 7)


def derive(run):
    """Add the derived finetext fields so METRICS can treat them uniformly."""
    ft = run.get("finetext")
    if not ft or "total_found" not in ft:
        return run
    correct = sum(ft.get(f"recall_{t}px", 0) for t in TIERS)
    ft["_correct"] = correct
    ft["_fabricated"] = ft["total_found"] - correct
    return run


def parse_arm(spec):
    """`label=pattern[,pattern...]`, or a bare tag.

    One arm is not always one tag family. run_engine_compare.sh writes REPEATS
    as a tag PREFIX (`<prefix><rep>_<model>_think<mode>`) while this file's own
    convention is a `-rep<N>` SUFFIX, so a repeated arm pools only via a glob --
    and a bare glob then becomes the column header of a published table. Worse,
    a campaign that pools runs from two families (the qwen3.8 ROCm n=5 arm pools
    `rocm-5d5b7a72-*` with `rocm-n3-*`) cannot be expressed as one glob at all;
    it was rendered by copying score files to matching names, by hand, which is
    the manipulation ADR 0012 rule 8 exists to keep out of a record.

    So an arm may name itself and list what it pools:

        summarize_reps.py 'think-on (n=5)=rocm-5d5b7a72-qwen38-thinkon,rocm-5d5b7a72-qwen38-thinkon-np4400,rocm-n3-qwen38-thinkon-rep*'
    """
    label, sep, pats = spec.partition("=")
    if not sep:
        return spec, [spec]
    # The label is everything before the FIRST "=", so it cannot contain one.
    # Nor should it want to: the "(n=N)" suffix is appended by the renderer, and
    # writing it into the label yields "think-on (n (n=4)" plus a first pattern
    # of "5)=..." that matches nothing -- a silently short arm, which is the one
    # failure mode a pooled column must not have. Caught by running it.
    parts = pats.split(",")
    # An "=" inside the label does not produce an empty pattern -- it produces a
    # plausible-looking one ("5)=rocm-...") that matches no file, so the arm is
    # dropped with a warning and the table renders one column short. A record
    # missing an arm looks like a record that only had one. Fatal, not a warning.
    if not pats or any(not q or "=" in q for q in parts):
        sys.exit(f"summarize_reps: bad arm spec {spec!r} -- a label may not "
                 f"contain '=' (the '(n=N)' suffix is added for you) and a "
                 f"pattern may not be blank")
    return label, parts


def load_arm(spec):
    label, pats = parse_arm(spec)
    paths = sorted({q for pat in pats
                    for q in glob.glob(os.path.join(DIR, f"scores_{pat}.json"))
                    + glob.glob(os.path.join(DIR, f"scores_{pat}-rep*.json"))})
    runs = []
    for q in paths:
        with open(q) as fh:
            runs.append(derive(json.load(fh)))
    return runs, paths


# Sections that carry their OWN window by design, not by escalation.
# vision_suite.py:1136 gives finetext `NUM_CTX or FINETEXT_NUM_CTX` (32768)
# where every other section takes `NUM_CTX or 16384`, so with NUM_CTX unset the
# two differ in every run ever measured. Folding that into the escalation
# warning made the warning fire always, and a warning that always fires carries
# no information -- the same failure as a self-check that can never pass. It is
# still worth reporting, so it is reported as what it is: an annotation, not an
# alarm.
OWN_WINDOW = ("finetext",)


def _window(runs, key):
    """The window this arm ran at, and whether it held still.

    Repeats of one arm are supposed to sit at one rung, so a section that moves
    between repeats means the arm escalated and its rows are not internally
    comparable. THAT is what the ⚠ marks. A section listed in OWN_WINDOW is
    excluded from it and annotated instead, because its difference is by design
    and says nothing about escalation.

    Requested value wins over served: files written before the num_ctx fold
    (#153) recorded the suite default in `num_ctx` for the finetext block while
    `req_num_ctx` kept the value actually asked for, so reading served-first
    would report 16384 for a block that ran at 32768.
    """
    def values(sections):
        seen = []
        for r in runs:
            for name, sec in r.items():
                if name in sections and isinstance(sec, dict):
                    v = sec.get(f"req_{key}") or sec.get(key)
                    if v is not None and v not in seen:
                        seen.append(v)
        return seen

    names = {n for r in runs for n in r}
    suite = values(names - set(OWN_WINDOW))
    own = values(names & set(OWN_WINDOW))

    if not suite:
        return "—" if not own else "/".join(str(v) for v in sorted(own))
    s = (str(suite[0]) if len(suite) == 1
         else "/".join(str(v) for v in sorted(suite)) + " ⚠")
    if own and own != suite:
        s += " (finetext " + "/".join(str(v) for v in sorted(own)) + ")"
    return s


def rung(runs):
    return _window(runs, "num_ctx")


def npred(runs):
    return _window(runs, "num_predict")


def cell(runs, section, key, kind):
    # A capped rep is an unfinished measurement (ADR 0012 conv 9) and must
    # not enter a pooled mean — this file pooled them for its whole life
    # because it never imported the one capped definition (SPEC H5, and the
    # conformance table's claim that it did was false at HEAD).
    secs = [r[section] for r in runs
            if section in r and key in r.get(section, {})]
    vals = [s[key] for s in secs if not was_capped(s)]
    if secs and not vals:
        return "capped", None
    if not vals:
        return "—", None
    if kind == "bool":
        hits = sum(1 for v in vals if v)
        return (f"{hits}/{len(vals)} ✅" if hits == len(vals)
                else f"{hits}/{len(vals)} ❌"), 0 if hits in (0, len(vals)) else 1
    lo, hi = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    spread = hi - lo
    if kind == "float":
        # The observed range, NOT "mean ±spread/2". A ± interval is symmetric
        # about what precedes it, so mean ±(hi-lo)/2 describes [lo, hi] only
        # when the mean happens to be the midrange. At n=2 it always is, which
        # is why this read correctly for a whole campaign and then did not: the
        # five think-on name_bbox runs
        # (../vision-campaign-2026-08-17-qwen38-rocm.md) rendered
        # "0.742 ±0.103", an interval of [0.639, 0.846] that EXCLUDES the
        # observed minimum of 0.631 -- the very value the document's argument
        # rests on -- and claims headroom above a maximum of 0.838 that no run
        # reached. Printing lo and hi cannot do that, and it is what the
        # docstring promised: "the range is what a reader can actually reason
        # about".
        s = f"{mean:.3f}" + (f" [{lo:.3f}–{hi:.3f}]" if spread else "")
    else:
        s = f"{mean:.1f}".rstrip("0").rstrip(".") + (f" [{lo}–{hi}]" if spread else "")
    return s, spread


def main():
    tags = sys.argv[1:]
    if not tags:
        print(__doc__)
        return 2

    arms = []
    for spec in tags:
        label, _ = parse_arm(spec)
        runs, paths = load_arm(spec)
        if not runs:
            print(f"# no scores files for {spec}", file=sys.stderr)
            continue
        arms.append((label, runs))
        # The provenance line names every file behind the column, so a pooled
        # arm can be checked against its sources without rerunning anything.
        print(f"# {label}: n={len(runs)}  " + ", ".join(os.path.basename(p) for p in paths),
              file=sys.stderr)

    print("| metric | " + " | ".join(f"{t} (n={len(r)})" for t, r in arms) + " |")
    print("|---" * (len(arms) + 1) + "|")
    # The CONTEXT-ladder rung comes first because it conditions everything under
    # it. ADR 0029 requires think-on tables to report it, SPEC H4a calls it a
    # result rather than plumbing, and ADR 0012 rule 8 says the reported num_ctx
    # is the final successful rung. Two arms measured at different rungs are not
    # comparable on tok/s or req/h at all — KV size drives decode speed — so a
    # table without this row invites exactly the comparison it cannot support.
    print(f"| **num_ctx rung** | " + " | ".join(rung(runs) for _, runs in arms) + " |")
    print(f"| **num_predict** | " + " | ".join(npred(runs) for _, runs in arms) + " |")

    for section, key, label, kind in METRICS:
        cells = [cell(runs, section, key, kind)[0] for _, runs in arms]
        if all(c == "—" for c in cells):
            continue
        print(f"| {label} | " + " | ".join(cells) + " |")

    # The point of repeating: how much does one arm move on its own?
    print()
    print("Within-arm spread (max-min), the bar any cross-arm claim must clear:")
    for tag, runs in arms:
        if len(runs) < 2:
            print(f"  {tag}: n={len(runs)}, no spread measurable")
            continue
        # Grouped by unit, because a single ranking across both is meaningless:
        # an IoU spread of 0.207 and a count spread of 2 are not comparable, and
        # sorting them together let four finetext counts fill a top-4 list while
        # the largest IoU movement in the arm went unprinted. A reader who wants
        # the IoU spread then reaches for it by hand, which is how the wrong
        # 0.103 reached the campaign record's prose.
        worst = {"float": [], "int": []}
        for section, key, label, kind in METRICS:
            if kind == "bool":
                continue
            _, sp = cell(runs, section, key, kind)
            if sp:
                worst[kind].append((sp, label))
        parts = []
        for kind, unit in (("float", "ratios"), ("int", "counts")):
            if not worst[kind]:
                continue
            worst[kind].sort(reverse=True)
            parts.append(f"{unit} — "
                         + ", ".join(f"{l} {s:g}" for s, l in worst[kind][:4]))
        if parts:
            print(f"  {tag}: " + "; ".join(parts))
        else:
            print(f"  {tag}: identical across all {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
