#!/usr/bin/env python3
"""Render the engine-comparison tables from scores_<tag>.json + ft_<tag>.json.

Usage:
    python3 summarize_engine_compare.py [--dir RUNDIR] [--think false|on] \
        <model> [model ...]

--think selects which think cell to render (default "false"). Both cells are
produced by run_engine_compare.sh and are separate results — render them as two
tables, never merged into one.

Models are the names given to run_engine_compare.sh, in row order; tags derive
the same way (':' and '.' become '_'). Output is the exact two-markdown-table
format of the 2026-08-08 MLX-vs-GGUF campaign
(../vision-campaign-2026-08-08-mlx.md) — keep it stable so runs diff cleanly.

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


def resolve_tag(rundir, model, think):
    """Prefer the think-suffixed tag; fall back to the legacy bare tag.

    The fallback applies ONLY to think=false. Pre-2026-08-09 runs wrote the
    bare tag and were think-off, so serving them for a --think on request
    would silently render the wrong cell — the caller would see think-off
    numbers labelled as reasoning results.
    """
    suffixed = tag_for(model, think)
    if os.path.exists(os.path.join(rundir, f"scores_{suffixed}.json")):
        return suffixed
    if think == "false":
        legacy = tag_for(model)
        if os.path.exists(os.path.join(rundir, f"scores_{legacy}.json")):
            return legacy
    return suffixed  # nothing on disk; report against the expected name


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
        v = (sec or {}).get("num_ctx")
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
    except OSError:
        return None


def fmt_bool(v):
    return "✅" if v else "❌"


def main():
    args = sys.argv[1:]
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
    if not args:
        sys.exit(__doc__)
    engine_map = {}
    for pair in os.environ.get("ENGINE_MAP", "").split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            engine_map[k.strip()] = v.strip()

    t1 = ["| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial "
          "| Invoice (items · qty+price · total) | name_bbox hits |",
          "|---|---|---|---|---|---|---|---|"]
    t2 = ["| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) "
          "| Gen tok/s | Prefill tok/s | s/req | req/h |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    for model in args:
        tag = resolve_tag(rundir, model, think)
        eng = engine_for(model, engine_map)
        eng_cell = f"**{eng}**" if eng == "MLX" else eng
        scores = load(os.path.join(rundir, f"scores_{tag}.json")) or {}
        ft = load(os.path.join(rundir, f"ft_{tag}.json")) or {}
        sc = scores.get("scene_single", {})
        dc = scores.get("document_single", {})
        mu = scores.get("multi_3img", {})

        # The window these numbers were achieved under. A cell measured at a
        # different num_ctx is not comparable on throughput (KV size affects
        # decode speed), and a short result at a small window may be a cap
        # rather than the model. "—" means the run predates the field.
        ctx_cell = ctx_for(sc, dc, mu, ft)

        iou = sc.get("bbox_mean_iou")
        iou_cell = "—" if iou is None else (f"**{iou:.3f}**" if eng == "MLX" else f"{iou:.3f}")
        blc = (f"{sc.get('bbox_hits', '—')}/{sc.get('object_count', '—')} · "
               f"{sc.get('labels_found', '—')}/{sc.get('labels_total', '—')} · "
               f"{sc.get('colors_right', '—')}/{sc.get('labels_total', '—')}")
        inv = (f"{dc.get('items_found', '—')}/{dc.get('items_total', '—')} · "
               f"{dc.get('qty_price_right', '—')}/{dc.get('items_total', '—')} · "
               f"{fmt_bool(dc.get('total_right'))}")
        t1.append(f"| {model} | {eng_cell} | {ctx_cell} | {iou_cell} | {blc} | "
                  f"{fmt_bool(sc.get('serial_found'))} | {inv} | {dc.get('name_bbox_hits', '—')} |")

        tiers = [str(ft.get(f"recall_{px}px", "—")) for px in (22, 16, 12, 9, 7)]
        if mu.get("q1_right") and mu.get("q2_right") and mu.get("q4_bbox_hit"):
            multi = "✅ all Qs + bbox"
        elif not mu:
            multi = "—"
        else:
            fails = [q for q in ("q1_right", "q2_right", "q4_bbox_hit") if not mu.get(q)]
            multi = "❌ " + ", ".join(fails)
        gen = sc.get("gen_tps")
        pre = sc.get("prefill_tps")
        # Unique-image steady state (baseline §4.2): decode + full prefill from
        # the scene run's clean rates; req/h = 3600 / s_req, serial.
        s_req = None
        if gen and pre and sc.get("eval_count") and sc.get("prompt_eval_count"):
            s_req = sc["eval_count"] / gen + sc["prompt_eval_count"] / pre
        s_cell = f"{s_req:.1f}" if s_req else "—"
        rh_cell = f"{3600 / s_req:.0f}" if s_req else "—"
        t2.append(f"| {model} | {eng_cell} | {ctx_cell} | " + " | ".join(tiers) +
                  f" | {multi} | {round(gen) if gen else '—'} | {round(pre) if pre else '—'}"
                  f" | {s_cell} | {rh_cell} |")

    print("## Scene grounding (six objects, norm-1000 boxes) + document extraction\n")
    print("\n".join(t1))
    print("\n## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput\n")
    print("\n".join(t2))


if __name__ == "__main__":
    main()
