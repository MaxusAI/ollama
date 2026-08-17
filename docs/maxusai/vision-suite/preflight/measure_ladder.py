#!/usr/bin/env python3
"""Measure a new (platform, arch) row and print it as paste-ready TOML.

This exists because the documented path was a heredoc in README.md that you
copy, edit for host and model, run, then read numbers off stdout and type into
expectations.toml. ADR 0012 rule 8 says never to transcribe generator output by
hand, and that rule was written after a hand-copied table published a scene IoU
of 0.000 that was really 0.872. The same exposure applies to a ladder: five
numbers, a prefix and two budgets, typed from a terminal into a config file.

So this prints the block. Copy it whole; do not retype any part of it.

    python3 measure_ladder.py --host http://127.0.0.1:11434 \
        --model qwen3.8:27b-q4_K_M --profile rocm-0-32-1-dynres --arch qwen35 \
        --container ollama-rocm

Nothing here decides whether a number is *correct* — that is what preflight is
for. Run preflight against the new row afterwards and expect it to pass; if it
does not, the row is wrong and must not be edited to make it green (ADR 0011).

Geometries come from expectations.toml's own `ladder_sizes`, never a local copy,
so this cannot drift from what the harness actually checks.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probes import (Ollama, ProbeError, container_logs,  # noqa: E402
                    parse_pixel_lines)

DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import tomllib
except ModuleNotFoundError:  # py<3.11
    import tomli as tomllib


def budgets_from_log(container, since, log_cmd=None):
    """Read the load_hparams pixel budget, the way payload_proof does.

    Returns (min_px, max_px, custom_bounds) or None. The bounds matter: the fork
    sets both for gemma4/nemotron_h_omni but only the min for the qwen VL family,
    whose ceiling is llama.cpp's structural set_limit_image_tokens(8, 4096). A
    row that claims a bound is ours when no flag sets it fails payload_proof.
    """
    try:
        lines = parse_pixel_lines(container_logs(container, since, log_cmd))
    except Exception as exc:
        print(f"  (could not read container logs: {exc})", file=sys.stderr)
        return None
    if not lines:
        return None
    got = {}
    for d in lines:
        got[d["kind"]] = d
    if "min" not in got or "max" not in got:
        return None
    custom = sorted(k for k in ("min", "max") if got[k]["custom"])
    return got["min"]["value"], got["max"]["value"], custom


TODO_STRIDE = "patch_stride = 0        # TODO patch_size * spatial_merge_size"
TODO_BUDGETS = [
    "budget_min_tokens = 0   # TODO from the load_hparams log",
    "budget_max_tokens = 0   # TODO from the load_hparams log",
    "image_min_pixels = 0    # TODO budget_min_tokens * patch_stride^2",
    "image_max_pixels = 0    # TODO budget_max_tokens * patch_stride^2",
]

# Deliberately NOT inferred. Several strides divide the same pixel counts --
# qwen35's 1048576/4194304 are divisible by both 16^2 and 32^2 -- and the wrong
# choice yields budgets that are arithmetically self-consistent and therefore
# invisible: test_verdicts asserts image_min_pixels == budget_min_tokens *
# stride^2, which holds for 4096*16^2 exactly as it does for the correct
# 1024*32^2, and payload_proof compares the pixel values, which are right either
# way. A guessed stride would ship a wrong row past every gate. Measured
# 2026-08-17 by running this tool against a known row.
NO_STRIDE = ("  patch_stride not given: budgets left as TODO. It is patch_size * "
             "spatial_merge_size from the model's config -- pass --stride; this "
             "tool will not guess it.")


def budget_fields(stride, budgets):
    """Decide the patch_stride/budget_*/image_* lines of the row.

    Pure: `budgets` is budgets_from_log's return, None when the log could not be
    read or held no complete pair. Returns (lines, notes) -- TOML for stdout,
    warnings for stderr -- so the decision can be tested without a server.
    """
    if not budgets:
        if not stride:
            return [TODO_STRIDE] + TODO_BUDGETS, [NO_STRIDE]
        # KEEP THE OPERATOR'S STRIDE. Emitting `patch_stride = 0  # TODO` here
        # would throw away a value that was typed on the command line and ask
        # for it to be typed again into the config file -- the hand-transcription
        # ADR 0012 rule 8 exists to prevent, and gratuitous, since nothing about
        # an unreadable log casts doubt on the stride.
        #
        # Nor is this a provenance step down: patch_stride is operator-asserted
        # on EVERY path, including the one below. The log never supplies it; it
        # supplies pixel counts that the stride is checked against. So the only
        # thing missing here is that check, which the note says plainly.
        return [f"patch_stride = {stride}"] + TODO_BUDGETS, [
            f"  no budget read: keeping patch_stride = {stride} from --stride, "
            f"budget fields left as TODO. Nothing cross-checked that stride -- "
            f"with a readable load_hparams line this tool divides the pixel "
            f"counts by it and refuses on a remainder, and that check did not "
            f"run here."]

    min_px, max_px, custom = budgets
    if not stride:
        return [TODO_STRIDE] + TODO_BUDGETS, [NO_STRIDE]

    bmin, bmax = min_px // (stride * stride), max_px // (stride * stride)
    # ROUND-TRIP GUARD. The division above is only meaningful if the pixel counts
    # and the stride describe THE SAME MODEL. Nothing so far establishes that:
    # budgets_from_log takes whatever load_hparams lines fall in the time window,
    # and a concurrent probe of another model puts ITS lines there.
    #
    # That is not hypothetical. Measuring qwen35 (stride 32) while a preflight
    # run was still probing gemma4 (stride 48) read gemma4's 161280/2580480 and
    # emitted:
    #
    #     budget_min_tokens = 157     image_min_pixels = 160768
    #     budget_max_tokens = 2520    image_max_pixels = 2580480
    #
    # A row nobody could look at and call wrong -- except that 157 * 32^2 =
    # 160768, not the 161280 it came from. Integer division lost the remainder,
    # and the remainder is the evidence. When the pixels really do belong to this
    # stride the division is exact, so an inexact one means the two halves came
    # from different models.
    #
    # This is the same class of defect as a guessed patch_stride, and it is worse
    # in one way: a guessed stride is at least self-consistent and this is not,
    # so it can be caught for free. Refuse rather than emit -- test_verdicts
    # asserts image_min_pixels == budget * stride^2 against whatever is written,
    # so a fabricated row that satisfies its own arithmetic would pass every gate
    # the repo has.
    if bmin * stride * stride != min_px or bmax * stride * stride != max_px:
        # The stride goes back to TODO here, unlike the no-log case above: a
        # remainder says the pixels and the stride disagree, and it cannot say
        # WHICH is wrong. A concurrent load is the likelier cause, but a mistyped
        # --stride produces the identical remainder. Echo the contested value in
        # the comment so it is not lost, and refuse to print it as fact.
        return [f"patch_stride = 0        # TODO -- --stride said {stride}, "
                f"which the log contradicts"] + TODO_BUDGETS, [
            f"  REFUSING the budget read: {min_px}/{max_px} pixels do not divide "
            f"exactly by stride^2 ({stride}^2={stride * stride}) -- "
            f"{bmin}*{stride}^2={bmin * stride * stride} and "
            f"{bmax}*{stride}^2={bmax * stride * stride}. The pixel lines and the "
            f"stride describe different models, which happens when another model "
            f"was loaded inside the log window; a mistyped --stride does the same. "
            f"Run this against an idle server and check nothing else is probing "
            f"it. Budgets and patch_stride left as TODO; the ladder below is "
            f"unaffected, being measured directly."]

    lines = [f"patch_stride = {stride}",
             f"budget_min_tokens = {bmin}",
             f"budget_max_tokens = {bmax}",
             f"image_min_pixels = {bmin * stride * stride}",
             f"image_max_pixels = {bmax * stride * stride}"]
    if custom is not None and set(custom) != {"min", "max"}:
        lines.append(f"custom_bounds = {json.dumps(custom)}"
                     f"   # only these are set by our flags")
    return lines, []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="e.g. http://127.0.0.1:11434")
    ap.add_argument("--model", required=True, help="the tag this row will name")
    ap.add_argument("--profile", required=True, help="e.g. rocm-0-32-1-dynres")
    ap.add_argument("--arch", required=True, help="e.g. qwen35")
    ap.add_argument("--container", help="read the pixel budget from this "
                                        "container's load log; without it the "
                                        "budget fields are left as TODO")
    ap.add_argument("--log-cmd", help="template for reading container logs")
    ap.add_argument("--stride", type=int, help="patch_size * spatial_merge_size; "
                                               "derived from the budget when the "
                                               "log is readable")
    ap.add_argument("--expectations", default=os.path.join(DIR, "expectations.toml"))
    ap.add_argument("--out", help="also write the raw measurement as JSON")
    args = ap.parse_args()

    with open(args.expectations, "rb") as fh:
        exp = tomllib.load(fh)
    sizes = exp["ladder_sizes"]
    tol = exp.get("ladder_tolerance", 2)

    c = Ollama(args.host)
    print(f"server:  {c.version()}", file=sys.stderr)
    print(f"model:   {args.model}", file=sys.stderr)

    since = time.time()
    if args.container:
        # Force a fresh load_hparams block, exactly as preflight does, so the
        # budget read below belongs to THIS model and not whatever ran before.
        print("  unloading to force a fresh load_hparams block...", file=sys.stderr)
        try:
            c.unload(args.model)
        except ProbeError as exc:
            print(f"  (unload failed, budget may be stale: {exc})", file=sys.stderr)

    prefix, text_only, detail = c.image_prefix(args.model)
    print(f"  prefix={prefix} text_only={text_only} {detail}", file=sys.stderr)

    ladder = []
    for size in sizes:
        delta, _ = c.visual_tokens(args.model, size, prefix)
        ladder.append(delta)
        print(f"  {size:>10}: {delta}", file=sys.stderr)

    budgets = budgets_from_log(args.container, since,
                               args.log_cmd) if args.container else None
    if args.container and not budgets:
        print("  no complete load_hparams pixel pair in the log window: the "
              "budget fields cannot be filled from this run.", file=sys.stderr)

    lines, notes = budget_fields(args.stride, budgets)
    for note in notes:
        print(note, file=sys.stderr)

    flat = len(set(ladder)) == 1
    print(file=sys.stderr)
    print("# ---- paste this into expectations.toml; do not retype any of it ----",
          file=sys.stderr)

    print(f"[expect.{args.profile}.{args.arch}]")
    print('status = "measured"')
    print(f'measured_on = "{time.strftime("%Y-%m-%d")}"')
    print(f'model = "{args.model}"')
    for line in lines:
        print(line)
    print(f'scaling = "{"flat" if flat else "dynamic"}"'
          + ("   # verify: flat is correct under 004, wrong for a dynres arch"
             if flat else ""))
    print(f"ladder_tolerance = {tol}")
    print(f"# prefix == {prefix}, text_only == {text_only}; measured with"
          f" image_prefix (B8), not text_baseline")
    print(f"ladder = {json.dumps(ladder)}")

    print(file=sys.stderr)
    print("Now add the arch to the profile's `arches` list -- an [expect.*] block",
          file=sys.stderr)
    print("alone is inert (ADR 0011 rule 4) -- and run preflight to confirm.",
          file=sys.stderr)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"host": args.host, "model": args.model,
                       "profile": args.profile, "arch": args.arch,
                       "prefix": prefix, "text_only": text_only,
                       "sizes": sizes, "ladder": ladder,
                       # Both halves of the budget arithmetic, so the raw file
                       # can be checked against the row that was pasted.
                       "stride": args.stride, "budgets": budgets}, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
