#!/usr/bin/env python3
"""Render the engine-comparison tables from scores_<tag>.json + ft_<tag>.json.

Usage:
    python3 summarize_engine_compare.py [--dir RUNDIR] [--think false|on] \
        [--prefix TAG_PREFIX] <model> [model ...]

--think selects which think cell to render (default "false"). Both cells are
produced by run_engine_compare.sh and are separate results — render them as two
tables, never merged into one.

Models are the names given to run_engine_compare.sh, in row order; tags derive
the same way (':' and '.' become '_'). Output is the exact two-markdown-table
format of the 2026-08-08 MLX-vs-GGUF campaign
(../vision-campaign-2026-08-08-mlx.md) — keep it stable so runs diff cleanly.

The one column that moves with --think is the token count, because the quantity
itself does: see `token_column`.

Engine column: safetensors tags are the MLX engine on this fork; the store
names them by MLX-side quantization ("-nvfp4"). Anything else renders GGUF.
Override per model with ENGINE_MAP="model=Engine,model=Engine" if a store
breaks that naming convention.
"""
import json
import os
import sys


def tag_for(model, think=None):
    """Tag for a model, optionally for a think mode.

    run_engine_compare.sh writes <model>_think<mode> so the two think cells
    cannot overwrite each other. Runs predating 2026-08-09 wrote the bare
    <model> tag; resolve_tag falls back to that so old score files still
    render.
    """
    base = model.replace(":", "_").replace(".", "_")
    return base if think is None else f"{base}_think{think}"


def resolve_tag(rundir, model, think, prefix=""):
    """Prefer the think-suffixed tag; fall back to the legacy bare tag.

    The fallback applies ONLY to think=false. Pre-2026-08-09 runs wrote the
    bare tag and were think-off, so serving them for a --think on request
    would silently render the wrong cell — the caller would see think-off
    numbers labelled as reasoning results.

    `prefix` is the campaign namespace run_engine_compare.sh writes when
    TAG_PREFIX is set — e.g. "cudafull1_" (TAG_PREFIX "cudafull" plus the
    interpolated rep and separator). Without it T1 could not render a
    prefixed campaign at all: the 2026-08-20 five-model cudafull1 baseline
    had only the T2 pivot until this landed. Empty by default (H4).
    """
    suffixed = prefix + tag_for(model, think)
    if os.path.exists(os.path.join(rundir, f"scores_{suffixed}.json")):
        return suffixed
    if think == "false":
        legacy = prefix + tag_for(model)
        if os.path.exists(os.path.join(rundir, f"scores_{legacy}.json")):
            return legacy
    return suffixed  # nothing on disk; report against the expected name


def was_capped(sec):
    """True when generation stopped at num_predict rather than finishing.

    A capped cell's eval_count IS the cap, so any throughput derived from it
    measures the harness setting, not the model — and it moves 2.3x when the
    ladder escalates a rung. Such a req/h must never be quoted as a model
    result.

    The server's own verdict wins where recorded: done_reason "length" is
    capped and "stop" is finished, whatever the token arithmetic says. The
    arithmetic misreads two real cases — a model that emits its final token
    exactly at the cap (eval == num_predict, done_reason "stop"), and a
    thinking continuation that overshoots the recorded cap (eval 8290 against
    8192, measured 2026-08-20 on qwen3.6 bbox_contract_reasoning). Blocks
    recorded before 2026-08-20 carry no done_reason and keep the
    eval_count >= num_predict fallback — as does a connection-closed final,
    which serializes with the field omitted (llm/server.go maps
    DoneReasonConnectionClosed to "" and the API field is omitempty).
    """
    sec = sec or {}
    dr = sec.get("done_reason")
    if dr == "length":
        return True
    if dr == "stop":
        return False
    # req_num_predict is the fallback cap spelling: legacy finetext blocks
    # recorded the suite default in num_predict while req_num_predict held
    # the real request. Reading both HERE keeps this the one definition —
    # two call sites had grown disagreeing aliasing shims around it.
    cap = sec.get("num_predict") or sec.get("req_num_predict")
    ev = sec.get("eval_count")
    return bool(cap and ev and ev >= cap)


def save(path, data):
    """Atomic JSON write: tmp + os.replace, the probes.py idiom. The scores
    file is the campaign's most expensive artifact; a truncate-then-dump
    writer that dies mid-dump destroys it."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)


def capped_arms(path, ctx_max=None, only=None):
    """Arms in a scores file that must ESCALATE: capped per was_capped, an
    escalatable error (window overflow, or a blown generation budget — both
    are cured by a bigger rung, since num_predict and the derived timeout
    scale with it), minus arms already marked NOT CONVERGED at a ceiling
    >= ctx_max, which have nothing left to learn below that ceiling.

    The campaign driver imports this instead of re-deriving "capped" (SPEC
    H5). The shell heredoc it replaces compared eval_count alone, and both
    misread classes are on record: a stop-overshoot (eval 8290 against 8192,
    done_reason "stop") escalated a finished cudafull1 cell, and a synthetic
    "length" below the cap — the fork's window-bound continuation — ended a
    cell with no NOT-CONVERGED verdict at all. Other errors are NOT
    escalation candidates: the request itself failed, and resume re-runs it
    at the same rung. Raising CTX_MAX above a recorded marker reopens the
    arm here AND at the driver's cell-level ceiling_standing skip.
    """
    data = load(path)
    if not data:
        return []
    out = []
    for name, blk in data.items():
        if only and name not in only:
            # ONLY_TESTS scopes the run: a stale capped block left by an
            # earlier, wider invocation of the same tag must not drive this
            # run's ladder nor collect a fabricated ceiling verdict for an
            # arm this campaign never attempted.
            continue
        if not isinstance(blk, dict):
            continue
        marker = blk.get("ladder_not_converged_at")
        if marker and ctx_max and int(marker) >= int(ctx_max):
            continue
        err = blk.get("error")
        if err is not None:
            if isinstance(err, str) and ("num_ctx too small" in err
                                         or "context overflow" in err
                                         or "blown generation budget" in err):
                out.append(name)
            elif was_capped(blk.get("prior") or {}):
                # The error replaced a CAPPED measurement, so the arm's last
                # known state is "this rung was not enough" and it still needs a
                # bigger one. Without this the arm left the ladder entirely: the
                # driver breaks its rung loop as soon as nothing is capped, so a
                # transport failure on a capped arm ended the cell as CONVERGED
                # with no NOT-CONVERGED marker. The rule above is unchanged for
                # an error with no capped history — that re-runs at the same
                # rung rather than buying a bigger window.
                out.append(name)
            continue
        if was_capped(blk):
            out.append(name)
    return out


def ceiling_standing(path, ctx_max, only=None):
    """True when this cell's ladder verdict already stands: at least one arm
    is marked NOT CONVERGED at a ceiling >= ctx_max and no arm still has
    work below it. The DRIVER consults this before starting a cell's ladder,
    so a resumed ceiling cell costs zero restarts and zero probe runs —
    while arm_done stays exactly SPEC H4b (capped always re-runs): reopening
    is the driver's decision, taken where CTX_MAX is known, not a resume
    exception inside the suite.
    """
    data = load(path)
    if not data:
        return False
    has_marked = False
    for name, blk in data.items():
        if only and name not in only:
            continue
        if not isinstance(blk, dict):
            continue
        if "error" in blk:
            return False        # a failed request always has work to do
        marker = blk.get("ladder_not_converged_at")
        if marker and int(marker) >= int(ctx_max):
            has_marked = True
            continue
        if was_capped(blk):
            return False        # capped without a sufficient ceiling verdict
    return has_marked


def mark_not_converged(path, arms, num_ctx):
    """Stamp the ceiling verdict into still-capped blocks (blueprint P0-2).

    NOT CONVERGED used to exist only as a stdout line, so a ceiling cell was
    byte-identical to a not-yet-escalated one and every resume re-climbed
    the whole ladder. The marker records the HIGHEST window that failed to
    converge (only ever raised — a later low-CTX_MAX run must not erase a
    proven ceiling), skips error blocks (a failed request is not a ladder
    verdict), and never raises: this runs under the driver's `set -eu` at
    the one point where the most inference has already been paid for, so an
    I/O failure logs and returns rather than killing the campaign.
    """
    try:
        data = load(path)
        if not data:
            return
        for a in arms:
            blk = data.get(a)
            if isinstance(blk, dict) and "error" not in blk:
                blk["ladder_not_converged_at"] = max(
                    int(blk.get("ladder_not_converged_at") or 0), int(num_ctx))
        save(path, data)
    except Exception as exc:
        print(f"mark_not_converged: {exc}", file=sys.stderr)


def ctx_for(*sections):
    """The num_ctx a row's results were measured at.

    Every section of a run shares one window, so they should agree. If they do
    not — e.g. vision_suite.py and finetext_probe.py were run without the
    runner setting both, and their differing defaults applied — render all
    distinct values rather than silently picking one, because the row then
    mixes windows and is not internally comparable.
    """
    seen = []
    for sec in sections:
        # Requested wins over served, for the same reason summarize_reps' _window
        # does it: files written before the num_ctx fold (#153) recorded the
        # SUITE default in num_ctx for the finetext block while req_num_ctx kept
        # what was actually asked for. Reading served-first reports those rows at
        # a window their finetext never ran at, and hides the mixed-window ⚠ this
        # function exists to raise.
        v = (sec or {}).get("req_num_ctx") or (sec or {}).get("num_ctx")
        if v is not None and v not in seen:
            seen.append(v)
    if not seen:
        return "—"
    return str(seen[0]) if len(seen) == 1 else "/".join(str(v) for v in sorted(seen)) + " ⚠"


def engine_for(model, engine_map):
    if model in engine_map:
        return engine_map[model]
    return "MLX" if any(k in model for k in ("nvfp4", "mlx", "mxfp8")) else "GGUF"


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        # ValueError covers a truncated/corrupt file (json.JSONDecodeError
        # subclasses it): the ladder helpers run under the driver's set -eu,
        # and "unreadable" must degrade to "no data", never kill a campaign.
        return None


def fmt_bool(v):
    return "✅" if v else "❌"


def token_column(think):
    """Header for the eval_count column, which is not the same quantity in both
    think modes.

    eval_count is EVERY token the model generated. With thinking off that is the
    answer, and "Answer tok" is exact. With it on the count is thinking + answer
    -- a think-on gemma4:12b row reads 5588 against answers of a few hundred
    tokens -- so the same header becomes a claim the run cannot support.

    The split is not available to fix it with. The API reports one count; the
    parser that knows where reasoning ends takes text, not tokens, so it has no
    count to report (#189). Renaming is the honest option, not a lesser one:
    "Gen tok" is what eval_count is in both modes, and it is what s/req and
    req/h are derived from, which is why those columns stay correct here while
    the label was wrong.

    Think-off keeps the narrower header. The runner asked for no reasoning and
    every column of that table has been published under it since 2026-08-08;
    widening it would rewrite a true label to guard against a case the mode
    excludes.
    """
    return "Answer tok" if think == "false" else "Gen tok"


def main():
    args = sys.argv[1:]
    # Ladder subcommands for run_engine_compare.sh: values travel as argv,
    # never spliced into python -c source (a quote in TAG_PREFIX or the
    # checkout path was a SyntaxError that killed campaigns under set -eu).
    if args and args[0] == "capped-arms":
        only = (set(args[3].split(",")) if len(args) > 3 and args[3] else None)
        print(" ".join(capped_arms(
            args[1], int(args[2]) if len(args) > 2 and args[2] else None,
            only)))
        return 0
    if args and args[0] == "mark-not-converged":
        mark_not_converged(args[1], args[3:], int(args[2]))
        return 0
    if args and args[0] == "ceiling-standing":
        only = (set(args[3].split(",")) if len(args) > 3 and args[3] else None)
        return 0 if ceiling_standing(args[1], int(args[2]), only) else 1
    rundir = os.path.dirname(os.path.abspath(__file__))
    if args and args[0] == "--dir":
        rundir = args[1]
        args = args[2:]
    # Which think cell to render. Both are produced by run_engine_compare.sh;
    # they are separate results and must not be mixed in one table.
    think = "false"
    if args and args[0] == "--think":
        think = args[1]
        args = args[2:]
    # Campaign tag namespace, matching run_engine_compare.sh's TAG_PREFIX
    # output (give the full literal prefix, e.g. "cudafull1_"). Unset, tags
    # derive exactly as before (H4).
    prefix = ""
    if args and args[0] == "--prefix":
        prefix = args[1]
        args = args[2:]
    if not args:
        sys.exit(__doc__)
    engine_map = {}
    for pair in os.environ.get("ENGINE_MAP", "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            engine_map[k.strip()] = v.strip()

    # Provenance, rendered from the score files instead of typed into the
    # report (ADR 0012 rule 1). Cells carry host/server_version since H11;
    # older runs render "pre-H11". Mixed values are flagged, not averaged: a
    # table whose rows ran on two hosts is two campaigns wearing one header.
    prov_hosts, prov_vers = set(), set()

    t1 = ["| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial "
          "| Invoice (items · qty+price · total) | name_bbox in-band |",
          "|---|---|---|---|---|---|---|---|"]
    t2 = ["| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) "
          f"| Multi anchored | Think tok | {token_column(think)} | Gen tok/s | Prefill tok/s | s/req | req/h |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    for model in args:
        tag = resolve_tag(rundir, model, think, prefix)
        eng = engine_for(model, engine_map)
        eng_cell = f"**{eng}**" if eng == "MLX" else eng
        scores = load(os.path.join(rundir, f"scores_{tag}.json")) or {}
        # Suite-produced finetext first; ft_<tag>.json only as the pre-fold
        # (1db8ec9c) fallback. See summarize_head_to_head.py for why.
        ft = scores.get("finetext") or load(os.path.join(rundir, f"ft_{tag}.json")) or {}
        sc = scores.get("scene_single", {})
        dc = scores.get("document_single", {})
        mu = scores.get("multi_3img", {})
        # The anchored variant is a SEPARATE column, never folded into the one
        # beside it — the pair is the whole evidence that a q4-bbox miss was a
        # frame error rather than a grounding failure (same argument as T2's
        # anchored row; measured 2026-08-20, qwen3.8 AND nemotron3 think-on
        # both flip ❌→✅ under the calibration entry, for different reasons).
        ma = scores.get("multi_3img_anchored", {})

        # The window these numbers were achieved under. A cell measured at a
        # different num_ctx is not comparable on throughput (KV size affects
        # decode speed), and a short result at a small window may be a cap
        # rather than the model. "—" means the run predates the field.
        ctx_cell = ctx_for(sc, dc, mu, ft, ma)
        for sec in (sc, dc, mu, ft, ma):
            if sec.get("host"):
                prov_hosts.add(sec["host"])
            if sec.get("server_version"):
                prov_vers.add(sec["server_version"])
        # A loaded pre-H11 file contributes the sentinel so mixing it with an
        # H11-era row renders both entries and trips the MIXED warning --
        # otherwise the footer shows one clean host for rows it can't vouch for.
        if scores and not any(sec.get("host") for sec in (sc, dc, mu, ft, ma)):
            prov_hosts.add("pre-H11 run (host not recorded)")
        if scores and not any(sec.get("server_version") for sec in (sc, dc, mu, ft, ma)):
            prov_vers.add("pre-H11 run (build not recorded)")

        def q(block, text):
            # ADR 0012 conv 9: a capped cell renders "capped", never a score.
            # T1 carries the rung in its num_ctx column, so no bracket here.
            # Guarding only the scene-derived throughput cells (below) left
            # every quality cell exposed — measured 2026-08-20, the capped
            # qwen3.6 think-on multi ceiling cell rendered
            # "❌ q1_right, q2_right, q4_bbox_hit": "cannot ground" for a cell
            # that never terminated, the exact defect fixed in T2 that morning.
            return "capped" if was_capped(block) else text

        iou = sc.get("bbox_mean_iou")
        iou_cell = "—" if iou is None else (f"**{iou:.3f}**" if eng == "MLX" else f"{iou:.3f}")
        blc = (f"{sc.get('bbox_hits', '—')}/{sc.get('object_count', '—')} · "
               f"{sc.get('labels_found', '—')}/{sc.get('labels_total', '—')} · "
               f"{sc.get('colors_right', '—')}/{sc.get('labels_total', '—')}")
        inv = (f"{dc.get('items_found', '—')}/{dc.get('items_total', '—')} · "
               f"{dc.get('qty_price_right', '—')}/{dc.get('items_total', '—')} · "
               f"{fmt_bool(dc.get('total_right'))}")
        t1.append(f"| {model} | {eng_cell} | {ctx_cell} | {q(sc, iou_cell)} | {q(sc, blc)} | "
                  f"{q(sc, fmt_bool(sc.get('serial_found')))} | {q(dc, inv)} | "
                  f"{q(dc, str(dc.get('name_bbox_hits', '—')))} |")

        tiers = [q(ft, str(ft.get(f"recall_{px}px", "—"))) for px in (22, 16, 12, 9, 7)]

        def multi_cell(block):
            # NOT "all Qs": the multi-image prompt asks four questions and q3
            # is never scored, so a cell that answered q3 wrongly still reads
            # as a clean sweep. Name the three that are actually gated.
            if was_capped(block):
                return "capped"
            if block.get("q1_right") and block.get("q2_right") and block.get("q4_bbox_hit"):
                return "✅ q1 + q2 + q4-bbox"
            if not block:
                return "—"
            fails = [k for k in ("q1_right", "q2_right", "q4_bbox_hit")
                     if not block.get(k)]
            return "❌ " + ", ".join(fails)

        multi = multi_cell(mu)
        anchored = multi_cell(ma)
        gen = sc.get("gen_tps")
        pre = sc.get("prefill_tps")
        # Unique-image steady state (baseline §4.2): decode + full prefill from
        # the scene run's clean rates; req/h = 3600 / s_req, serial.
        s_req = None
        if gen and pre and sc.get("eval_count") and sc.get("prompt_eval_count"):
            s_req = sc["eval_count"] / gen + sc["prompt_eval_count"] / pre
        # Generated length is half the throughput story: a slower model that
        # finishes in fewer tokens can beat a faster, more verbose one, since
        # s/req = eval/gen_tps + prompt_eval/prefill_tps. Surface it directly.
        # Under think-on those tokens include the reasoning, which is why the
        # header follows the mode -- see token_column.
        ev = sc.get("eval_count")
        # Exact reasoning-token count from token_split.py, which tokenizes the
        # persisted thinking text with the server's own vocab and proves the
        # split against eval_count (its acceptance gate). Written into the
        # scene block by --write; absent means the split has not run or the
        # gate refused a tokenizer for this model — render "—", never a
        # char-proportional estimate (a 62-token response measured 21% control
        # tokens, so proportions lie). Think-off rows show a true 0.
        tt = sc.get("thinking_tokens")
        if was_capped(sc):
            # eval_count is the cap here, not a generated length; both it and
            # anything derived from it are meaningless as model results.
            tok_cell = f"≥{ev} ⚠"
            think_cell = "capped"
            s_cell = "capped"
            rh_cell = "capped"
        else:
            tok_cell = str(ev) if ev else "—"
            think_cell = str(tt) if tt is not None else "—"
            s_cell = f"{s_req:.1f}" if s_req else "—"
            rh_cell = f"{3600 / s_req:.0f}" if s_req else "—"
        t2.append(f"| {model} | {eng_cell} | {ctx_cell} | " + " | ".join(tiers) +
                  f" | {multi} | {anchored} | {think_cell} | {tok_cell} | {round(gen) if gen else '—'} | {round(pre) if pre else '—'}"
                  f" | {s_cell} | {rh_cell} |")

    print("## Scene grounding (six objects, norm-1000 boxes) + document extraction\n")
    print("\n".join(t1))
    print("\n## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput\n")
    print("\n".join(t2))

    hosts = sorted(prov_hosts) or ["pre-H11 run (host not recorded)"]
    vers = sorted(prov_vers) or ["pre-H11 run (build not recorded)"]
    warn = " ⚠ MIXED — rows are not one campaign" if len(prov_hosts) > 1 or len(prov_vers) > 1 else ""
    print(f"\nProvenance (from score files): host(s) {', '.join(hosts)} · "
          f"build(s) {', '.join(vers)} · think={think}{warn}")


if __name__ == "__main__":
    # sys.exit propagates main()'s return: ceiling-standing answers via exit
    # code, and a discarded return read as "standing" at EVERY ctx_max —
    # caught by the CLI smoke, invisible to function-level tests.
    sys.exit(main())
