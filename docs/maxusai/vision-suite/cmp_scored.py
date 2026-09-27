#!/usr/bin/env python3
"""Which scored fields moved between two score files, test by test, and which way.

    python3 cmp_scored.py A.json B.json [tol]

Counting every scalar field (the run directory's cmp_scores.py) is right for "are two runs identical", but a change
that only moves thinking lengths then reads as hundreds of differing cells. This prints only the quality fields, with
the side each move favours, for the tests that finished on both sides, then the tests that finished on one side
only. A capped block (was_capped, SPEC H5) and an errored one are not scores. Numeric moves within tol (default
0.005, the IoU band of ADR 0029) are not moves. Labels (declared type, bbox space ...) print when they change and
favour neither side.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_engine_compare import load, was_capped  # noqa: E402  (H5: import, never redefine)

HIGHER = {"bbox_hits", "bbox_mean_iou", "name_bbox_hits", "name_bbox_mean_iou", "hits_anchor", "hits_bestfit",
          "hits_declared", "iou_anchor", "iou_declared", "iou_at_implied_scale", "labels_found", "items_found",
          "chart_values_found", "colors_right", "total_found", "total_right", "q1_right", "q2_right",
          "qty_price_right", "q4_bbox_hit", "serial_found", "invoice_no", "recall_7px", "recall_9px", "recall_12px",
          "recall_16px", "recall_22px", "json_valid", "contract_followed", "declaration_valid",
          "declaration_matches_boxes"}
LOWER = {"degenerate_boxes"}
LABELS = {"declared_type", "declared_ref", "declared_order", "declaration_scope", "bbox_space", "name_bbox_space",
          "q4_bbox_space", "bestfit_dialect", "field_name", "offered_key", "anchor_implied_ref",
          "anchor_implied_type", "anchor_present", "anchor_beats_declared", "implied_scale", "object_count",
          "self_check"}


def scored(block):
    return isinstance(block, dict) and not block.get("error") and not was_capped(block)


def _num(v):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) else None


def compare(a, b, tol=0.005):
    """Return (lines, moves favouring A, moves favouring B, label changes) for two score dicts."""
    lines, only, fav_a, fav_b, labels = [], [], 0, 0, 0
    for name in sorted(set(a) | set(b)):
        ba, bb = a.get(name), b.get(name)
        sa, sb = scored(ba), scored(bb)
        if sa != sb:
            only.append(f"{name}: finished only in {'A' if sa else 'B'}")
        if not (sa and sb):
            continue
        moves = []
        for k in sorted((HIGHER | LOWER | LABELS) & (set(ba) | set(bb))):
            va, vb = ba.get(k), bb.get(k)
            if va == vb:
                continue
            na, nb = _num(va), _num(vb)
            if k in LABELS or na is None or nb is None:
                moves.append(f"{k} {va!r}->{vb!r}")
                labels += 1
                continue
            if abs(nb - na) <= tol + 1e-9:  # 0.968 - 0.963 is a hair over 0.005 in floats
                continue
            favours_b = (nb > na) == (k in HIGHER)
            fav_b += favours_b
            fav_a += not favours_b
            moves.append(f"{k} {va!r}->{vb!r} {'B+' if favours_b else 'A+'}")
        if moves:
            lines.append(f"{name}: " + "; ".join(moves))
    return lines + only, fav_a, fav_b, labels


def main(argv):
    if len(argv) not in (3, 4):
        sys.exit(__doc__)
    a, b = load(argv[1]), load(argv[2])
    if a is None or b is None:
        sys.exit(f"unreadable score file: {argv[1] if a is None else argv[2]}")
    lines, fav_a, fav_b, labels = compare(a, b, float(argv[3]) if len(argv) == 4 else 0.005)
    for line in lines:
        print(line)
    print(f"quality moves: {fav_b} favour B, {fav_a} favour A, {labels} label changes "
          f"(A = {os.path.basename(argv[1])}, B = {os.path.basename(argv[2])})")


if __name__ == "__main__":
    main(sys.argv)
