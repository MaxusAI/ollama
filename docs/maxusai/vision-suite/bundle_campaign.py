#!/usr/bin/env python3
"""Bundle a campaign's score files into the one JSON file that is committed.

    python3 bundle_campaign.py [--dir RUNDIR] --out bench-runs/vision-campaign-<date>-<label>.json \\
        [--meta-file META.json] [--meta KEY=VALUE ...] \\
        --set MODELS="<model> ..." [THINK_MODES="false on"] [TAG_PREFIX=<p>] [REPEATS=<n>] \\
        [--set ...]

A bundle is {"meta": {...}, "cells": [{"tag", "scores", "finetext_probe"}, ...]}. Each
cell is one run's scores_<tag>.json and ft_<tag>.json, copied unchanged;
finetext_probe is null when the run left no ft file. `summarize_engine_compare.py
--bundle` renders a campaign's tables from the bundle, so the tables a document
publishes re-render from the repo alone (ADR 0012 rule 8, SPEC H7). --dir defaults to
this directory, where run_engine_compare.sh writes its score files.

SETS. Each --set names cells the way run_engine_compare.sh was told to run them,
using the runner's own knobs: MODELS, THINK_MODES (default "false on", as in the
runner), TAG_PREFIX and REPEATS. Pass the knobs of each runner invocation as one --set.
- Cells are bundled in the runner's order: set by set, then model, think mode and
  repeat.
- Tags come from arm_prefix and resolve_tag in summarize_engine_compare.py, which
  build on tag_for, as the summarizers do (SPEC H5, H6). So the bundler cannot name a
  cell differently from the tables that read it.
- Other runner knobs (ONLY_TESTS, RESTART_CMD, ...) do not change a tag, and are
  refused. A pasted invocation cannot stand in for a set it is not.

REFUSALS.
- A missing or unreadable scores file is an error, not an empty cell. A bundle that
  silently lacks a row records a campaign that did not happen.
- Two sets that name the same tag are refused.
- A (model, think) cell in DESCOPED_CELLS is skipped and reported, because the runner
  never measures it.
- A cell is never modified: it is the parsed file, serialized again.

META comes from --meta-file (a JSON object, key order kept), then from each --meta
KEY=VALUE in order. Three keys are checked:
- `host` and `server_version` are the two stamps every score block carries (SPEC
  H11). If the meta gives one, it must be the value every cell recorded. Leave it out
  when a campaign spans several hosts or builds by design.
- `host_profile` (SPEC H26) must name a file beside the bundle. Without one the
  bundler warns.

The bundle is written with json.dump(indent=1) and a trailing newline, atomically
(`save`), so an interrupted write never leaves half a bundle. Nothing is written when
any check fails.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# SPEC H5: the tag rule, the descoping policy and the JSON readers and writers are
# imported from the one module that defines them, never redefined here.
from summarize_engine_compare import arm_prefix, is_descoped, load, resolve_tag, save  # noqa: E402

KNOBS = ("MODELS", "THINK_MODES", "TAG_PREFIX", "REPEATS")
STAMPS = ("host", "server_version")        # written into every score block (SPEC H11)


class BundleError(Exception):
    """A bundle that would misstate the run. main() reports it and exits 1."""


def parse_set(tokens):
    """One --set's KNOB=VALUE tokens, as the runner reads them."""
    knobs = {}
    for tok in tokens:
        key, sep, value = tok.partition("=")
        if not sep:
            raise BundleError(f"--set {tok!r}: expected KNOB=VALUE")
        if key not in KNOBS:
            raise BundleError(f"--set {key}: only {', '.join(KNOBS)} decide a cell's tag, "
                              "so only they belong in a set")
        if key in knobs:
            raise BundleError(f"--set gives {key} twice")
        knobs[key] = value
    models = knobs.get("MODELS", "").split()
    if not models:
        raise BundleError("--set needs MODELS")
    think_modes = knobs.get("THINK_MODES", "false on").split()      # the runner's default
    if not think_modes:
        raise BundleError("--set THINK_MODES is empty")
    try:
        repeats = int(knobs.get("REPEATS", "1"))
    except ValueError:
        raise BundleError(f"--set REPEATS={knobs['REPEATS']!r} is not a whole number") from None
    if repeats < 1:
        raise BundleError("--set REPEATS must be at least 1")
    return {"models": models, "think_modes": think_modes,
            "tag_prefix": knobs.get("TAG_PREFIX", ""), "repeats": repeats}


def read_cell_file(path):
    """A run file's parsed JSON. `load` reads an unreadable file as None, which is
    right for the ladder and wrong here: a bundle must not record a broken file as
    an empty one."""
    data = load(path)
    if not isinstance(data, dict):
        raise BundleError(f"{path}: unreadable, or not a JSON object")
    return data


def build_cells(rundir, sets):
    """The cells the sets name, in the runner's order, each read unchanged."""
    cells, seen = [], set()
    for spec in sets:
        for model in spec["models"]:
            for think in spec["think_modes"]:
                if is_descoped(model, think):
                    print(f"skipped {model} think={think}: descoped by policy "
                          "(DESCOPED_CELLS), so the runner never measured it", file=sys.stderr)
                    continue
                for rep in range(1, spec["repeats"] + 1):
                    prefix = arm_prefix(spec["tag_prefix"], rep, spec["repeats"])
                    tag = resolve_tag(rundir, model, think, prefix)
                    if tag in seen:
                        raise BundleError(f"{tag} is named by two sets")
                    seen.add(tag)
                    path = os.path.join(rundir, f"scores_{tag}.json")
                    if not os.path.exists(path):
                        raise BundleError(f"{path}: no scores file for {model} think={think}")
                    probe = os.path.join(rundir, f"ft_{tag}.json")
                    cells.append({"tag": tag, "scores": read_cell_file(path),
                                  "finetext_probe": (read_cell_file(probe)
                                                     if os.path.exists(probe) else None)})
    return cells


def recorded(cells, key):
    """Every value the cells' blocks recorded for one H11 stamp."""
    values = set()
    for cell in cells:
        for block in list(cell["scores"].values()) + [cell["finetext_probe"]]:
            if isinstance(block, dict) and block.get(key):
                values.add(block[key])
    return values


def check_meta(meta, cells, out):
    """Refuse a meta the cells contradict, or one that names a missing host profile."""
    for key in STAMPS:
        if key not in meta:
            continue
        values = recorded(cells, key)
        if values - {meta[key]}:
            raise BundleError(f"meta {key} is {meta[key]!r}, but the cells recorded "
                              f"{', '.join(sorted(values))}. Correct it, or leave {key} out "
                              "of the meta when the campaign spans several")
    profile = meta.get("host_profile")
    if profile is None:
        print("warning: the meta names no host_profile. A published campaign commits the "
              "profile of the machine it ran on beside its bundle (SPEC H26)", file=sys.stderr)
    elif not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(out)), profile)):
        raise BundleError(f"meta host_profile {profile!r} is not beside the bundle "
                          f"({os.path.dirname(os.path.abspath(out))}); SPEC H26")


def read_meta(meta_file, pairs):
    """--meta-file's object, then each --meta KEY=VALUE, in order."""
    meta = {}
    if meta_file:
        data = load(meta_file)
        if not isinstance(data, dict):
            raise BundleError(f"{meta_file}: unreadable, or not a JSON object")
        meta.update(data)
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise BundleError(f"--meta {pair!r}: expected KEY=VALUE")
        meta[key] = value
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="where the run's scores_<tag>.json and ft_<tag>.json are "
                         "(default: this directory, where the runner writes them)")
    ap.add_argument("--out", required=True, help="the bundle to write")
    ap.add_argument("--meta-file", help="a JSON object: the meta, in key order")
    ap.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                    help="a meta entry; applied after --meta-file, in order")
    ap.add_argument("--set", action="append", nargs="+", required=True, dest="sets",
                    metavar="KNOB=VALUE",
                    help="one runner invocation's MODELS, THINK_MODES, TAG_PREFIX, REPEATS")
    args = ap.parse_args(argv)
    try:
        sets = [parse_set(tokens) for tokens in args.sets]
        meta = read_meta(args.meta_file, args.meta)
        cells = build_cells(args.dir, sets)
        check_meta(meta, cells, args.out)
    except BundleError as exc:
        print(f"bundle_campaign: {exc}", file=sys.stderr)
        return 1
    save(args.out, {"meta": meta, "cells": cells}, end="\n")
    print(f"{len(cells)} cells -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
