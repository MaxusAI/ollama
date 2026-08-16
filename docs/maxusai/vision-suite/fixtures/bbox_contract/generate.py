#!/usr/bin/env python3
"""Build the bbox_contract fixture corpus from ground truth.

The fixtures are DERIVED, not captured. Each one reconstructs a response shape
the scorer has to handle, computed from visimgs/ground_truth.json so the numbers
are exact and the construction is auditable. They are not model output and must
not be cited as measurement — see README.md.

Run from this directory:  python3 generate.py
"""
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))
GT = json.load(open(os.path.join(DIR, "..", "..", "visimgs", "ground_truth.json")))
SCENE = GT["scene_hd"]
W, H = SCENE["size"]
OBJ = [(o["label"], o["bbox"]) for o in SCENE["objects"]]


def dump(group, name, payload):
    """Write a fixture as raw response text."""
    d = os.path.join(DIR, group)
    os.makedirs(d, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=1)
    with open(os.path.join(d, f"{name}.txt"), "w") as f:
        f.write(text)


def objects(fn, field="box_2d", labels=None):
    return [{"label": lab, field: fn(b)}
            for lab, b in OBJ if labels is None or lab in labels]


xyxy = lambda b: list(b)
yxyx = lambda b: [b[1], b[0], b[3], b[2]]
norm1000 = lambda b: [round(b[0] / W * 1000), round(b[1] / H * 1000),
                      round(b[2] / W * 1000), round(b[3] / H * 1000)]
norm1 = lambda b: [round(b[0] / W, 4), round(b[1] / H, 4),
                   round(b[2] / W, 4), round(b[3] / H, 4)]
scaled = lambda k: (lambda b: [round(v * k) for v in b])

# --- pre-existing shapes -------------------------------------------------
# Top-level declaration, positional arrays, no anchor. These are the responses
# the additive change promised not to disturb, so they carry the guarantee.

dump("preexisting", "clean_real_xyxy", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(xyxy)})

# gemma4 GGUF, measured 2026-08-16: declares xyxy, emits yxyx. Perfect vision,
# false self-description — the case a fixed-dialect scorer calls a bare miss.
dump("preexisting", "declared_xyxy_emitted_yxyx", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(yxyx)})

# nemotron GGUF: declares real, omits ref_size entirely, emits norm-1000.
dump("preexisting", "declared_real_no_refsize_emitted_norm1000", {
    "bbox_type": "real", "coord_order": "xyxy",
    "objects": objects(norm1000)})

# qwen3.8 GGUF: declares norm1000, emits real pixels in a ~1.33x frame.
dump("preexisting", "declared_norm1000_emitted_real_133x", {
    "bbox_type": "norm1000", "coord_order": "xyxy",
    "objects": objects(scaled(1.33))})

# qwen3.8 GGUF DYNAMO: uniform 1.30x inflation. Raw IoU collapses, but the
# shape was found — this is what implied_scale exists to record.
dump("preexisting", "uniform_scale_130x", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(scaled(1.30))})

dump("preexisting", "norm1_clean", {
    "bbox_type": "norm1", "coord_order": "xyxy",
    "objects": objects(norm1)})

dump("preexisting", "norm1000_clean", {
    "bbox_type": "norm1000", "coord_order": "xyxy",
    "objects": objects(norm1000)})

# Field-name dialects get_bbox accepts, one fixture each.
dump("preexisting", "field_bbox_2d", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(xyxy, field="bbox_2d")})

dump("preexisting", "field_bbox", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(xyxy, field="bbox")})

# Declaration absent: the scorer must not invent one.
dump("preexisting", "no_declaration", {"objects": objects(xyxy)})

# Only four of six located.
dump("preexisting", "partial_labels", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(xyxy, labels={"ANCHOR", "BEACON", "CIPHER", "DYNAMO"})})

# ref_size naming a frame that was never sent (qwen3.6 MLX).
dump("preexisting", "refsize_frame_never_sent", {
    "bbox_type": "real", "ref_size": [1024, 768], "coord_order": "xyxy",
    "objects": objects(xyxy)})

# Fenced output — parse_json_response has to strip the fence.
dump("preexisting", "fenced_json", "```json\n" + json.dumps({
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": objects(xyxy)}, indent=1) + "\n```")

# Unparseable: json_valid False and an early return, every other field default.
dump("preexisting", "malformed_json",
     "Here are the shapes I found in the image:\n\n"
     '{"bbox_type": "real", "objects": [{"label": "ANCHOR", "box_2d": [140, 160,')

# --- new-feature shapes --------------------------------------------------
# These exercise what 9c4416e5 added. The old scorer predates the syntax, so it
# scores them differently BY DESIGN; they are pinned as goldens, not compared.

dump("new_features", "perobject_declarations", {
    "objects": [{"label": lab, "bbox_type": "real", "coord_order": "xyxy",
                 "ref_size": [W, H], "box_2d": list(b)} for lab, b in OBJ]})

dump("new_features", "anchored_real", {
    "bbox_type": "real", "ref_size": [W, H], "coord_order": "xyxy",
    "objects": [{"label": "IMAGE", "box_2d": [0, 0, W, H]}] + objects(xyxy)})

# Anchor states a frame the declaration does not: the anchor is what recovers
# the true space, which is the point of scoring hits_anchor separately.
dump("new_features", "anchor_disagrees_with_declaration", {
    "bbox_type": "real", "ref_size": [1024, 768], "coord_order": "xyxy",
    "objects": [{"label": "IMAGE", "box_2d": [0, 0, W, H]}] + objects(xyxy)})

dump("new_features", "named_coords", {
    "bbox_type": "real", "ref_size": [W, H],
    "objects": [{"label": lab, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                for lab, b in OBJ]})

print("wrote fixtures under", DIR)
