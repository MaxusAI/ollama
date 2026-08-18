#!/usr/bin/env python3
"""Render the head-to-head pivot table (template T2, ADR 0012) from scores files.

Usage:
    python3 summarize_head_to_head.py [--dir RUNDIR] [--tags] <model-or-tag> ...

Rows are test × metric; columns are models — the deep-comparison complement to
summarize_engine_compare.py's wide format (T1). Use T2 for <= 4 models where
per-metric contrast matters; use T1 for full campaign matrices. Keep the row
set stable so reports diff cleanly.

By default arguments are model names and tags derive as run_engine_compare.sh
does (':' and '.' -> '_'). With --tags, arguments are literal score-file tags
(for ad-hoc runs whose tags are not model-derived).

Latency: s/req is the unique-image steady state (scene decode + full prefill
at clean rates — baseline report §3); req/h = 3600 / s_req, serial.
"""
import json
import os
import sys


def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError:
        return None


def b(v):
    return "✅" if v else "❌"


def ctx(block):
    """ADR 0012: quality cells carry the num_ctx they were measured at, in
    brackets. num_ctx is per model AND per test, and it changes what a result
    MEANS — an empty response is "truncated" at one window and "would not
    terminate" at another. A bare number hides which. "?" marks runs recorded
    before req_num_ctx existed."""
    # Falls back to num_ctx rather than declaring the window unknown. The
    # finetext block can arrive from the pre-fold ft_<tag>.json path, which
    # never wrote req_num_ctx, so reading only that field marks a cell "(?)"
    # whose window the very same block records — the marker then means "old
    # file OR fallback source" and stops identifying either.
    n = (block or {}).get("req_num_ctx") or (block or {}).get("num_ctx")
    return f" ({n})" if n else " (?)"


def main():
    args = sys.argv[1:]
    rundir = os.path.dirname(os.path.abspath(__file__))
    if args and args[0] == "--dir":
        rundir, args = args[1], args[2:]
    literal = False
    if args and args[0] == "--tags":
        literal, args = True, args[1:]
    if not args:
        sys.exit(__doc__)

    cols, cells = [], []
    for name in args:
        tag = name if literal else name.replace(":", "_").replace(".", "_")
        s = load(os.path.join(rundir, f"scores_{tag}.json")) or {}
        # Prefer the suite's own finetext block; fall back to the standalone
        # probe's ft_<tag>.json for runs recorded before the fold. Reading only
        # ft_ silently dropped real measurements: run_grid.sh never calls the
        # standalone probe, so its fine-text row rendered "—" even though
        # scores_<tag>.json held the tiers.
        ft = s.get("finetext") or load(os.path.join(rundir, f"ft_{tag}.json")) or {}
        sc, dc, mu = (s.get(k, {}) for k in ("scene_single", "document_single", "multi_3img"))
        gen, pre = sc.get("gen_tps"), sc.get("prefill_tps")
        s_req = (sc["eval_count"] / gen + sc["prompt_eval_count"] / pre
                 if gen and pre and sc.get("eval_count") and sc.get("prompt_eval_count") else None)
        doc_iou = dc.get("name_bbox_mean_iou")
        tiers = [ft.get(f"recall_{px}px") for px in (22, 16, 12, 9, 7)]
        cols.append(name)
        cells.append({
            ("scene", "bbox IoU"): (f"{sc['bbox_mean_iou']:.3f}" + ctx(sc)) if sc.get("bbox_mean_iou") is not None else "—",
            ("scene", "labels / serial"):
                f"{sc.get('labels_found', '—')}/{sc.get('labels_total', '—')}, {b(sc.get('serial_found'))}",
            ("document", "items / qty+price / total / invoice"):
                (f"{dc.get('items_found', '—')}/{dc.get('items_total', '—')}, "
                 f"{dc.get('qty_price_right', '—')}/{dc.get('items_total', '—')}, "
                 f"{b(dc.get('total_right'))}, {b(dc.get('invoice_no'))}") if dc else "—",
            ("document", "name_bbox IoU"): (f"{doc_iou:.3f}" + ctx(dc)) if doc_iou is not None else "—",
            ("fine text", "22/16/12/9/7 px"):
                ("/".join("—" if t is None else str(t) for t in tiers) + ctx(ft)) if ft else "—",
            ("multi (3 img)", "q1 / q2 / q4-bbox / chart"):
                (f"{b(mu.get('q1_right'))} {b(mu.get('q2_right'))} {b(mu.get('q4_bbox_hit'))} "
                 f"{mu.get('chart_values_found', '—')}/{mu.get('chart_total', '—')}" + ctx(mu)) if mu else "—",
            ("throughput", "gen tok/s"): f"{gen:.0f}" if gen else "—",
            ("throughput", "prefill tok/s"): f"{pre:.0f}" if pre else "—",
            ("latency", "s/req (unique image)"): f"{s_req:.1f}" if s_req else "—",
            ("latency", "req/h (serial)"): f"{3600 / s_req:.0f}" if s_req else "—",
        })

    rows = list(cells[0].keys())
    print("| test | metric | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for r in rows:
        print(f"| {r[0]} | {r[1]} | " + " | ".join(c[r] for c in cells) + " |")


if __name__ == "__main__":
    main()
