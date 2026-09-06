#!/usr/bin/env python3
"""Render the bbox-contract matrix across a campaign, plus its power-mode
provenance.

This is the generator behind the contract tables in the campaign documents. It
existed as an ad-hoc scratchpad script that produced published tables while
never being committed — exactly what SPEC H8 and ADR 0028 forbid, since the
numbers were citable but the code that made them was not reviewable.

Shared helpers come from summarize_engine_compare (SPEC H5); nothing here
redefines engine detection, the capped test, num_ctx extraction or tag
construction.

Usage:
  summarize_contract_matrix.py --think false <model> [<model> ...]
  summarize_contract_matrix.py --think on --log /tmp/fullsuite.log <model> ...
  summarize_contract_matrix.py --think on --prefix cudafull1_ <model> ...

--prefix is run_engine_compare.sh's TAG_PREFIX campaign namespace, the same
flag summarize_engine_compare.py and summarize_head_to_head.py already take.
Without it a prefixed campaign renders every arm as "—" and reads as a model
that answered nothing, when the scores are simply under another tag.

--log adds the per-model power-mode table. run_engine_compare.sh stamps
powermode into its run log per cell; blocks written since 2026-08-29 also
carry it (client.capture_stamps), but historical campaigns predate the field,
so provenance for an old campaign still has to come from the log (ADR 0012
rule 1).
"""
import os
import re
import sys

from summarize_engine_compare import (ctx_for, engine_for, load, resolve_tag,
                                      was_capped)

DIR = os.path.dirname(os.path.abspath(__file__))
ARMS = ["bbox_contract", "bbox_contract_multi", "bbox_contract_reasoning",
        "bbox_contract_pinned", "bbox_contract_perobject", "bbox_contract_anchored",
        "bbox_contract_adv_real", "bbox_contract_adv_norm1"]
SHORT = {a: a.replace("bbox_contract", "bc").replace("_", "") for a in ARMS}


def powermodes(logpath):
    """{(model, think): [powermode per rung]} from a runner log.

    A model-mode appears more than once when the num_ctx ladder escalates, and
    the rungs can straddle a power change — so keep every value rather than the
    first, and let the caller see a mixed row for what it is.
    """
    pm = {}
    for line in open(logpath, errors="replace"):
        m = re.match(r"##### MODEL (\S+) think=(\S+).*powermode=(\S+)", line)
        if m:
            pm.setdefault((m.group(1), m.group(2)), []).append(m.group(3))
    return pm


def main():
    argv = sys.argv[1:]
    think, logpath, prefix = "false", None, ""
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--think":
            think = argv.pop(0)
        elif flag == "--log":
            logpath = argv.pop(0)
        elif flag == "--prefix":
            prefix = argv.pop(0)
        else:
            sys.exit(f"unknown flag {flag}")
    models = argv
    if not models:
        sys.exit(__doc__)

    print(f"## Contract matrix (`contract_followed`), think={think}\n")
    print("| Model | Engine | " + " | ".join(SHORT[a] for a in ARMS) + " | num_ctx |")
    print("|---" * (len(ARMS) + 3) + "|")
    for model in models:
        tag = resolve_tag(DIR, model, think, prefix)
        d = load(os.path.join(DIR, f"scores_{tag}.json"))
        if d is None:
            print(f"| {model} | {engine_for(model, {})} | " +
                  " | ".join("—" for _ in ARMS) + " | — |")
            continue
        cells = []
        for a in ARMS:
            sec = d.get(a)
            if not sec:
                cells.append("—")
            elif "error" in sec:
                # An arm that ran and errored (OOM, transport, HTTP 500) has no
                # contract to judge. It rendered ❌ for the Sep-4 cudaMallocAsync
                # aborts and read as two contract failures on 31b (2026-09-06).
                cells.append("error")
            elif was_capped(sec):
                # ADR 0012 rule 8: a capped cell has no score to report. Say so
                # rather than rendering a False that reads as a measured failure.
                cells.append("cap")
            else:
                cells.append("✅" if sec.get("contract_followed") else "❌")
        eng = engine_for(model, {})
        print(f"| {model} | {'**MLX**' if eng == 'MLX' else eng} | " +
              " | ".join(cells) + f" | {ctx_for(*[d.get(a) for a in ARMS])} |")
    print("\n`error` = the arm ran and errored (OOM, transport, HTTP 500), so there is "
          "no contract to judge. "
          "`cap` = generation stopped at the `num_predict` cap rather than "
          "finishing, so the cell carries no score (ADR 0012 rule 8). The cap is "
          "a separate limit from the `num_ctx` window.")

    if logpath:
        pm = powermodes(logpath)
        print(f"\n## Power-mode provenance (from `{os.path.basename(logpath)}`)\n")
        print("| Model | think | powermode per rung |")
        print("|---|---|---|")
        for (model, mode), vals in sorted(pm.items()):
            if mode != think:
                continue
            flag = "" if len(set(vals)) == 1 else "  ⚠ mixed"
            print(f"| {model} | {mode} | {', '.join(vals)}{flag} |")
        print("\nMultiple values = the num_ctx ladder escalated. A mixed row "
              "straddles a power change: quality is power-invariant at "
              "temperature 0, throughput is not.")


if __name__ == "__main__":
    main()
