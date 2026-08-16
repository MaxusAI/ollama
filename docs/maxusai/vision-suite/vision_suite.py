#!/usr/bin/env python3
"""Nemotron vision test suite: long prompts, single & multi image, JSON + bboxes.
Usage: vision_suite.py <host> <tag> [model]
e.g.   vision_suite.py http://127.0.0.1:11435 patched nemotron3:33b-q4_K_M
"""
import json
import re, sys, base64, os, urllib.request

from sampling import sampling_for, provenance

HOST = TAG = MODEL = None  # set in main()
DIR = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(DIR, "visimgs")
GT = json.load(open(f"{IMG}/ground_truth.json"))

def b64(name):
    return base64.b64encode(open(f"{IMG}/{name}", "rb").read()).decode()

# Single source of truth for the request window, so what gets recorded in the
# scores cannot drift from what was actually sent. num_predict and the prompt
# share num_ctx; a cell whose eval_count equals num_predict was capped, and one
# where prompt + eval approaches num_ctx was bounded by the window instead.
def default_num_ctx():
    return int(os.environ.get("NUM_CTX", "16384"))


def default_num_predict():
    return int(os.environ.get("NUM_PREDICT", "2200"))


def _context_error(e, num_predict, num_ctx):
    """Turn the server's context-overflow 400 into an actionable message.

    The invariant is prompt + num_predict <= num_ctx, and the server enforces it
    before generating. Raising NUM_PREDICT without raising NUM_CTX therefore does
    not relieve truncation, it converts it into a bare `HTTP Error 400: Bad
    Request` that reads as a model or payload fault. It is neither — it is a
    harness misconfiguration, and this says so with the numbers needed to fix it."""
    try:
        body = e.read().decode()
    except Exception:
        body = ""
    m = re.search(r"n_prompt_tokens\\?[\"']?:\s*(\d+).*?n_ctx\\?[\"']?:\s*(\d+)", body, re.S)
    if e.code == 400 and ("exceed_context_size" in body or m):
        if m:
            need, ctx = int(m.group(1)), int(m.group(2))
            return RuntimeError(
                f"num_ctx too small: request needs {need} tokens but num_ctx is {ctx}. "
                f"prompt + num_predict must fit num_ctx (this call used "
                f"num_predict={num_predict}, num_ctx={num_ctx}). "
                f"Raise NUM_CTX to at least {need + 2048} (leaving headroom) — "
                f"raising NUM_PREDICT alone cannot fix this.")
        return RuntimeError(
            f"context overflow with num_predict={num_predict}, num_ctx={num_ctx}: {body[:200]}")
    return RuntimeError(f"HTTP {e.code}: {body[:200]}")


def gen(prompt, images, num_predict=None, num_ctx=None):
    if num_ctx is None:
        num_ctx = default_num_ctx()
    if num_predict is None:
        num_predict = default_num_predict()
    # Sampling is per-model and per-think-mode, NOT a hardcoded temperature 0.
    # Think-off is still greedy (every published baseline depends on that);
    # think-on uses the model card's values, because greedy decoding is what
    # made reasoning fail to terminate. See sampling.py and
    # ../runaway-reasoning-under-think.md.
    think_on = os.environ.get("THINK", "false") == "on"
    opts = {"num_predict": num_predict, "num_ctx": num_ctx}
    opts.update(sampling_for(MODEL, think_on))
    payload = {
        "model": MODEL, "prompt": prompt, "images": images,
        "stream": False, "format": "json",
        "options": opts,
    }
    if os.environ.get("KV_CACHE_TYPE"):
        payload["options"]["kv_cache_type"] = os.environ["KV_CACHE_TYPE"]
    # Fork-only per-request vision budget (visionServerArgs in llm/llama_server.go,
    # arch-gated to gemma4 and nemotron_h_omni). Pinning these to upstream's
    # effective defaults turns a fork build into a BUDGET-MATCHED CONTROL, which is
    # the only way to separate "our larger token budget changed the result" from
    # "the llama.cpp payload differs" when comparing against a stock server on a
    # different LLAMA_CPP_VERSION. See the control-arm section in README.md.
    # These are Runner options — changing them reloads the model.
    for env, opt in (("IMAGE_MIN_TOKENS", "image_min_tokens"),
                     ("IMAGE_MAX_TOKENS", "image_max_tokens")):
        if os.environ.get(env):
            payload["options"][opt] = int(os.environ[env])
    if not think_on:
        payload["think"] = False
    endpoint = os.environ.get("ENDPOINT", "generate")
    if endpoint == "chat":
        payload["messages"] = [{"role": "user", "content": payload.pop("prompt"),
                                "images": payload.pop("images")}]
        req = urllib.request.Request(HOST + "/api/chat",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=int(os.environ.get("HTTP_TIMEOUT", "1800"))))
        msg = r.get("message") or {}
        r["response"] = msg.get("content", "")
        r["thinking"] = msg.get("thinking", "")
        r["_num_predict"], r["_num_ctx"] = num_predict, num_ctx
        return r
    req = urllib.request.Request(HOST + "/api/generate",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=int(os.environ.get("HTTP_TIMEOUT", "1800"))))
    except urllib.error.HTTPError as e:
        raise _context_error(e, num_predict, num_ctx) from None
    # Stamp the EFFECTIVE limits so the caller records what actually ran rather
    # than what it thought it asked for — gen_opts, env and defaults all feed in.
    r["_num_predict"], r["_num_ctx"] = num_predict, num_ctx
    return r

SCENE_PROMPT = """You are a precision visual inspection system deployed in an industrial
quality-assurance pipeline. Your task on this frame is exhaustive object detection,
label transcription, and localization. Accuracy requirements are strict: downstream
robotic actuators consume your bounding boxes directly, so a box that misses its object
by more than a few percent of the frame causes a physical pick failure; a mis-transcribed
label causes the wrong part to be routed. Work methodically: first scan the entire frame
edge to edge, including corners and margins, then enumerate every distinct colored shape
you can find. For every shape, read the text label printed immediately above it — labels
are short uppercase code words, transcribe them EXACTLY, character by character, without
guessing or normalizing. If a label is genuinely illegible at the available resolution,
set "label" to null and "label_legible" to false rather than inventing a word; invented
labels are the single most damaging failure mode in this pipeline. Also transcribe any
other text present anywhere in the frame, however small, in the "other_text" array —
serial numbers, watermarks, footers, anything. Bounding boxes use ABSOLUTE PIXEL
coordinates in the original image coordinate system, formatted [x1, y1, x2, y2] where
(x1, y1) is the top-left corner and (x2, y2) the bottom-right corner of the shape itself
(not including its label text). The image is exactly {w} pixels wide and {h} pixels tall,
so all coordinates must lie in that range. For color, report the closest common English
color name (red, blue, green, orange, purple, teal, yellow, pink, brown, gray, black).
For shape kind use exactly "rectangle" or "ellipse". Respond with a SINGLE JSON object,
no prose before or after, following exactly this schema:
{{
  "image_width": <int>, "image_height": <int>,
  "object_count": <int>,
  "objects": [
    {{"label": <string or null>, "label_legible": <bool>, "kind": "rectangle"|"ellipse",
      "color": <string>, "bbox": [x1, y1, x2, y2], "confidence": <float 0..1>}}
  ],
  "other_text": [<string>, ...],
  "notes": <string, one short sentence on anything ambiguous>
}}
Do not omit any object. Do not merge adjacent objects. Count carefully before writing
object_count and make it equal to the length of the objects array."""

DOC_PROMPT = """You are an automated accounts-payable document parser. The attached image
is a scanned supplier invoice. Extract its contents COMPLETELY and EXACTLY into JSON for
direct ingestion into an ERP system; every field is compared against the purchase-order
database, so transcription must be verbatim — do not round numbers, do not paraphrase
item names, do not reformat identifiers. Read the entire page including headers, the
line-item table, totals, and any fine print at the bottom; fine-print reference codes
are mandatory fields for reconciliation. If any character is genuinely unreadable,
represent it as '?' rather than guessing. Amounts are in dollars; parse them as numbers
(strip the $ sign and thousands separators). For each line item give the bounding box of
the item-name text in ABSOLUTE PIXEL coordinates [x1, y1, x2, y2] (top-left and
bottom-right of the text run). The page is {w}x{h} pixels. Respond with a SINGLE JSON
object, no prose, exactly this schema:
{{
  "supplier": <string>, "invoice_number": <string>, "date": <string>,
  "customer": <string>,
  "line_items": [
    {{"name": <string>, "qty": <int>, "unit_price": <number>, "name_bbox": [x1,y1,x2,y2]}}
  ],
  "total": <number>,
  "fine_print": <string>,
  "all_reference_codes": [<string>, ...]
}}"""

MULTI_PROMPT = """You are a multi-document visual analyst. You receive THREE images in
order: image 1, image 2, image 3. Analyze each independently and then answer
cross-image questions. Be exhaustive but never invent content; if something is
unreadable, say so via null values rather than guessing. All bounding boxes are
ABSOLUTE PIXEL coordinates [x1, y1, x2, y2] in each image's own coordinate system.
For each image produce: a "type" classification (one of "shapes_scene",
"invoice_document", "bar_chart", "photo", "other"), a one-sentence "summary", a
"text_found" array with every distinct text string you can read in that image, and
"key_objects" — for a shapes scene: each shape with label+color+bbox; for a document:
the document identifier and the total amount; for a chart: every bar with its category
label and numeric value. Then answer the cross-image questions in the "answers" object:
q1: which image (1, 2 or 3) contains the reference code "INV-2026-0801"?
q2: in the bar chart, which category has the LARGEST value, and what is that value?
q3: does any single word that appears in image 1 also appear in image 2 or image 3?
    Answer with the word or null.
q4: give the bounding box, in image 1 pixel coordinates, of the shape whose label is
    "DYNAMO" (null if no such shape is legible).
Respond with a SINGLE JSON object, no prose:
{{
  "images": [
    {{"index": 1, "type": ..., "summary": ..., "text_found": [...], "key_objects": [...]}},
    {{"index": 2, ...}},
    {{"index": 3, ...}}
  ],
  "answers": {{"q1": <int>, "q2": {{"category": <string>, "value": <number>}},
               "q3": <string or null>, "q4": [x1,y1,x2,y2] or null}}
}}"""

def center_in(pred, gtb):
    try:
        cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
        return gtb[0] <= cx <= gtb[2] and gtb[1] <= cy <= gtb[3]
    except Exception:
        return False

def get_bbox(o):
    # Models speak different schema dialects: qwen-vl grounding uses "bbox_2d".
    for k in ("bbox", "bbox_2d", "box_2d"):
        if o.get(k):
            return o[k]
    return []


FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)

def parse_json_response(text):
    """json.loads with markdown-fence tolerance: engines that do not enforce
    format:"json" (the MLX runner before x/structured, ADR 0009) emit fenced
    JSON; stripping is a no-op on grammar-constrained output. Scorers record
    fenced=True so a non-enforcing engine is identifiable in the scores.
    Returns (obj_or_None, fenced)."""
    m = FENCE.search(text)
    if m:
        try:
            return json.loads(m.group(1)), True
        except Exception:
            return None, True
    try:
        return json.loads(text), False
    except Exception:
        return None, False

def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def score_scene(resp_text):
    g = GT["scene_hd"]
    W, H = g["size"]
    s = {"json_valid": False, "labels_found": 0, "labels_total": len(g["objects"]),
         "bbox_hits": 0, "bbox_mean_iou": 0.0, "bbox_space": None,
         "colors_right": 0, "serial_found": g["serial"] in resp_text,
         "object_count": None}
    r, fenced = parse_json_response(resp_text)
    if fenced:
        s["fenced"] = True
    if r is None:
        return s
    s["json_valid"] = True
    objs = r.get("objects") or []
    s["object_count"] = len(objs)
    by_label = {o.get("label"): o for o in objs if o.get("label")}
    matched = []
    for gto in g["objects"]:
        o = by_label.get(gto["label"])
        if o:
            s["labels_found"] += 1
            if (o.get("color") or "").lower() == gto["color"]:
                s["colors_right"] += 1
            bb = get_bbox(o)
            if len(bb) == 4:
                matched.append((bb, gto["bbox"]))
    # Models emit boxes in different coordinate spaces regardless of prompt
    # instructions (qwen3.6: 0-1000 normalized; nemotron w/ reasoning: pixels).
    # Score both spaces and keep the better one — report which.
    best = (0, 0.0, None)
    for space, fx, fy in (("pixel", 1.0, 1.0), ("norm1000", W/1000.0, H/1000.0)):
        for order in ("xyxy", "yxyx"):  # gemma4/Gemini box_2d is [y1,x1,y2,x2]
            hits, ious = 0, []
            for bb, gtb in matched:
                x1, y1, x2, y2 = (bb[0], bb[1], bb[2], bb[3]) if order == "xyxy" else (bb[1], bb[0], bb[3], bb[2])
                px = [x1*fx, y1*fy, x2*fx, y2*fy]
                hits += center_in(px, gtb)
                ious.append(iou(px, gtb))
            mean_iou = round(sum(ious)/len(ious), 3) if ious else 0.0
            if (mean_iou, hits) > (best[1], best[0]):
                best = (hits, mean_iou, f"{space}/{order}")
    s["bbox_hits"], s["bbox_mean_iou"], s["bbox_space"] = best
    if not s["serial_found"]:
        s["serial_found"] = g["serial"] in json.dumps(r)
    return s

def score_doc(resp_text):
    g = GT["document"]
    W, H = g["size"]
    s = {"json_valid": False, "invoice_no": False, "items_found": 0,
         "items_total": len(g["items"]), "qty_price_right": 0, "total_right": False,
         "name_bbox_hits": 0, "name_bbox_mean_iou": 0.0, "name_bbox_space": None}
    r, fenced = parse_json_response(resp_text)
    if fenced:
        s["fenced"] = True
    if r is None:
        return s
    s["json_valid"] = True
    s["invoice_no"] = g["invoice_no"] in json.dumps(r)
    items = r.get("line_items") or []
    matched = []
    for gti, gtb in zip(g["items"], g.get("name_bboxes") or [None] * len(g["items"])):
        m = next((i for i in items if isinstance(i.get("name"), str)
                  and gti["name"].lower() in i["name"].lower()), None)
        if m:
            s["items_found"] += 1
            try:
                if int(m.get("qty")) == gti["qty"] and abs(float(m.get("unit_price")) - gti["unit_price"]) < 0.01:
                    s["qty_price_right"] += 1
            except Exception:
                pass
            bb = m.get("name_bbox") or m.get("name_bbox_2d") or []
            if len(bb) == 4 and bb[1] > 250 and bb[3] < 700 and bb[0] < 500:
                s["name_bbox_hits"] += 1
            if len(bb) == 4 and gtb:
                matched.append((bb, gtb))
    # name_bbox_hits is a coarse band test that cannot see a 5% scale error —
    # it hid the document's vertical degradation for the whole 2026-08 bbox
    # investigation (findings doc §6). Score a real IoU against the measured
    # row geometry too, best-of-4 decode like score_scene.
    best = (0.0, None)
    for space, fx, fy in (("pixel", 1.0, 1.0), ("norm1000", W / 1000.0, H / 1000.0)):
        for order in ("xyxy", "yxyx"):
            ious = []
            for bb, gtb in matched:
                x1, y1, x2, y2 = (bb[0], bb[1], bb[2], bb[3]) if order == "xyxy" else (bb[1], bb[0], bb[3], bb[2])
                ious.append(iou([x1 * fx, y1 * fy, x2 * fx, y2 * fy], gtb))
            mean_iou = round(sum(ious) / len(ious), 3) if ious else 0.0
            if mean_iou > best[0]:
                best = (mean_iou, f"{space}/{order}")
    s["name_bbox_mean_iou"], s["name_bbox_space"] = best
    try:
        s["total_right"] = abs(float(r.get("total")) - g["total"]) < 0.01
    except Exception:
        pass
    return s

def score_multi(resp_text):
    g = GT
    s = {"json_valid": False, "q1_right": False, "q2_right": False,
         "q4_bbox_hit": False, "chart_values_found": 0,
         "chart_total": len(g["chart"]["bars"])}
    r, fenced = parse_json_response(resp_text)
    if fenced:
        s["fenced"] = True
    if r is None:
        return s
    s["json_valid"] = True
    a = r.get("answers") or {}
    s["q1_right"] = a.get("q1") == 2
    q2 = a.get("q2") or {}
    try:
        s["q2_right"] = str(q2.get("category", "")).strip().rstrip("*").upper() == "Q4" \
            and abs(float(q2.get("value")) - 128) < 0.5
    except Exception:
        pass
    dyn = next(o for o in g["scene_hd"]["objects"] if o["label"] == "DYNAMO")
    # q4 gets the same coordinate-dialect tolerance as scene boxes (score_scene):
    # models answer in their native space (norm-1000) regardless of the prompt.
    q4 = a.get("q4") or []
    if isinstance(q4, list) and len(q4) == 4:
        W, H = g["scene_hd"]["size"]
        try:
            for space, fx, fy in (("pixel", 1.0, 1.0), ("norm1000", W/1000.0, H/1000.0)):
                for order in ("xyxy", "yxyx"):
                    x1, y1, x2, y2 = (q4[0], q4[1], q4[2], q4[3]) if order == "xyxy" else (q4[1], q4[0], q4[3], q4[2])
                    if center_in([x1*fx, y1*fy, x2*fx, y2*fy], dyn["bbox"]):
                        s["q4_bbox_hit"] = True
                        s["q4_bbox_space"] = space + "/" + order
                        raise StopIteration
        except StopIteration:
            pass
        except Exception:
            pass
    blob = json.dumps(r)
    for b in g["chart"]["bars"]:
        if str(b["value"]) in blob:
            s["chart_values_found"] += 1
    return s


# --- bbox contract probe -----------------------------------------------------
#
# A NEW test, not an edit of scene_single. score_scene deliberately tries every
# coordinate dialect and keeps the best, because models answer in their native
# space whatever the prompt says. That tolerance is right for measuring
# grounding, but it makes one boolean do two jobs: a hit can mean "located it
# correctly" or "located it correctly AND obeyed the requested space", and the
# two are indistinguishable afterwards. Measured on qwen3.8 2026-08-16: MLX put
# DYNAMO's centre within 2px of truth but answered normalized-1000 when asked
# for pixels, and scored the same tick as an engine that had obeyed.
#
# Here the model DECLARES its space instead, so grounding is scored in the space
# it named — no guessing — and the declaration is scored separately. Either
# field name is accepted: gemma4/Gemini are trained toward box_2d and qwen-vl
# toward bbox_2d, so demanding one measures naming compliance, not vision. Which
# name the model chose is recorded instead.
#
# Kept as its own test so every historical scores_*.json stays comparable; the
# committed fine-text assets exist for the same reason.
BBOX_CONTRACT_PROMPT = """You are a localization service. Find every distinct
coloured shape in this image and report where each one is.

Declare the convention you used, then follow it exactly.

"bbox_type" — one of:
  - "real":     coordinates in pixels. You MUST also give "ref_size": [W, H],
                the width and height of the image those pixels refer to. If you
                resized the image internally, give the size YOU used, not the
                original.
  - "norm1":    coordinates scaled to 0.0-1.0 on both axes
  - "norm1000": coordinates scaled to 0-1000 on both axes

"coord_order" — one of:
  - "xyxy": [x1, y1, x2, y2]          (qwen-vl convention)
  - "yxyx": [y1, x1, y2, x2]          (gemma/Gemini box_2d convention)

No convention is preferred; pick the one you are most reliable in. But a box
that disagrees with your own declaration is worse than no answer at all,
because a consumer trusts the declaration.

Each box covers the shape itself, not its label text. Name the box field
"box_2d" or "bbox_2d", whichever is natural to you.

Respond with a SINGLE JSON object, no prose:
{{
  "bbox_type": "real" | "norm1" | "norm1000",
  "ref_size": [W, H],
  "coord_order": "xyxy" | "yxyx",
  "objects": [{{"label": "<uppercase code word above the shape>",
                "box_2d": [ , , , ]}}]
}}"""


def score_bbox_contract(resp_text):
    """Grounding and instruction-following, scored separately.

    Field names follow ms-swift's grounding schema (bbox_type real/norm1) so a
    passing response is directly usable as fine-tuning data; norm1000 is added
    because that is what gemma/Gemini emit and what ms-swift converts to for
    qwen3-vl.

    ref_size exists because "real" is meaningless on its own. Qwen-VL's absolute
    coordinates are relative to the RESIZED image, not the original, and a model
    can invent a scale outright: measured 2026-08-16, qwen3.8 GGUF returned
    DYNAMO at a uniform 1.30x the truth box — raw IoU 0.079, but 0.909 once
    divided by that factor. The shape was found; only the frame was wrong. A
    scorer that tries a fixed set of dialects reports that as a clean miss,
    which is how it was first mis-read as a grounding failure. implied_scale
    records the factor instead.
    """
    g = GT["scene_hd"]
    W, H = g["size"]
    s = {"json_valid": False, "declared_type": None, "declared_order": None,
         "declared_ref": None, "field_name": None,
         "labels_found": 0, "labels_total": len(g["objects"]),
         "hits_declared": 0, "iou_declared": 0.0,
         "hits_bestfit": 0, "bestfit_dialect": None,
         "implied_scale": None, "iou_at_implied_scale": None,
         "declaration_valid": False, "declaration_matches_boxes": False,
         "contract_followed": False}
    r, fenced = parse_json_response(resp_text)
    if fenced:
        s["fenced"] = True
    if r is None:
        return s
    s["json_valid"] = True

    btype = (r.get("bbox_type") or "").strip().lower()
    order = (r.get("coord_order") or "").strip().lower()
    ref = r.get("ref_size")
    s["declared_type"], s["declared_order"] = btype or None, order or None
    if isinstance(ref, list) and len(ref) == 2:
        s["declared_ref"] = [ref[0], ref[1]]
    # "real" is only a complete declaration with a reference size.
    s["declaration_valid"] = (
        order in ("xyxy", "yxyx")
        and (btype in ("norm1", "norm1000")
             or (btype == "real" and s["declared_ref"] is not None)))

    objs = r.get("objects") or []
    for o in objs:
        for k in ("box_2d", "bbox_2d", "bbox"):
            if o.get(k):
                s["field_name"] = k
                break
        if s["field_name"]:
            break

    by_label = {o.get("label"): o for o in objs if o.get("label")}
    matched = []
    for gto in g["objects"]:
        o = by_label.get(gto["label"])
        if o:
            s["labels_found"] += 1
            bb = get_bbox(o)
            if len(bb) == 4:
                matched.append((bb, gto["bbox"]))
    if not matched:
        return s

    def factors(btype, ref):
        if btype == "norm1":
            return W, H
        if btype == "norm1000":
            return W / 1000.0, H / 1000.0
        if btype == "real" and ref:
            try:
                return W / float(ref[0]), H / float(ref[1])
            except Exception:
                return 1.0, 1.0
        return 1.0, 1.0

    def tally(btype, order, ref=None):
        fx, fy = factors(btype, ref)
        hits, ious = 0, []
        for bb, gtb in matched:
            x1, y1, x2, y2 = ((bb[0], bb[1], bb[2], bb[3]) if order == "xyxy"
                              else (bb[1], bb[0], bb[3], bb[2]))
            px = [x1 * fx, y1 * fy, x2 * fx, y2 * fy]
            hits += center_in(px, gtb)
            ious.append(iou(px, gtb))
        return hits, round(sum(ious) / len(ious), 3) if ious else 0.0

    if s["declaration_valid"]:
        s["hits_declared"], s["iou_declared"] = tally(btype, order, s["declared_ref"])

    best = (0, 0.0, None)
    for bt in ("real", "norm1", "norm1000"):
        for od in ("xyxy", "yxyx"):
            hits, mi = tally(bt, od)          # "real" here means the native frame
            if (hits, mi) > (best[0], best[1]):
                best = (hits, mi, f"{bt}/{od}")
    s["hits_bestfit"], s["bestfit_dialect"] = best[0], best[2]

    # Diagnostic: if the boxes are the right SHAPE in the wrong FRAME, recover
    # the uniform factor rather than reporting a bare miss.
    od = order if order in ("xyxy", "yxyx") else "xyxy"
    ratios = []
    for bb, gtb in matched:
        x1, y1, x2, y2 = ((bb[0], bb[1], bb[2], bb[3]) if od == "xyxy"
                          else (bb[1], bb[0], bb[3], bb[2]))
        for pv, tv in zip((x1, y1, x2, y2), gtb):
            if tv:
                ratios.append(pv / tv)
    if ratios:
        k = sum(ratios) / len(ratios)
        if k > 0:
            s["implied_scale"] = round(k, 3)
            ious = []
            for bb, gtb in matched:
                x1, y1, x2, y2 = ((bb[0], bb[1], bb[2], bb[3]) if od == "xyxy"
                                  else (bb[1], bb[0], bb[3], bb[2]))
                ious.append(iou([x1 / k, y1 / k, x2 / k, y2 / k], gtb))
            s["iou_at_implied_scale"] = round(sum(ious) / len(ious), 3)

    s["declaration_matches_boxes"] = bool(
        s["declaration_valid"] and s["hits_declared"] == len(matched))
    s["contract_followed"] = bool(
        s["declaration_matches_boxes"] and s["hits_declared"] >= s["hits_bestfit"])
    return s



# The cross-image reasoning CONTROL. Same three images as bbox_contract_multi,
# but the model must USE the other two to answer q1/q2 before enumerating image
# one. It was added expecting to be the reproducer, on the strength of an
# ad-hoc run under the earlier two-field schema; under the current schema it
# passes 3/3 on qwen3.8 GGUF while bbox_contract_multi fails 3/3.
#
# Kept because that contrast is the finding: engaging with the distractors
# preserves an honest declaration, being told to ignore them does not. It
# carries the full six-box enumeration rather than one box, since a single box
# scores 0 or 1 and cannot separate a frame error from a miss.
BBOX_CONTRACT_REASONING_PROMPT = """You are given THREE images. Study all three,
then answer.

First, two questions that require comparing the images:
  q1: which image contains a bar chart, and how many bars does it have?
  q2: which image contains an invoice, and what is its invoice number?

Then, from the FIRST image only, find every distinct coloured shape and report
where each one is.

Declare the convention you used for those boxes, then follow it exactly.

"bbox_type" — one of:
  - "real":     coordinates in pixels. You MUST also give "ref_size": [W, H],
                the width and height of the image those pixels refer to. If you
                resized the image internally, give the size YOU used.
  - "norm1":    coordinates scaled to 0.0-1.0 on both axes
  - "norm1000": coordinates scaled to 0-1000 on both axes

"coord_order" — "xyxy" for [x1, y1, x2, y2], or "yxyx" for [y1, x1, y2, x2].

No convention is preferred. But a box that disagrees with your own declaration
is worse than no answer at all, because a consumer trusts the declaration.

Respond with a SINGLE JSON object, no prose:
{{
  "answers": {{"q1": <string>, "q2": <string>}},
  "bbox_type": "real" | "norm1" | "norm1000",
  "ref_size": [W, H],
  "coord_order": "xyxy" | "yxyx",
  "objects": [{{"label": "<uppercase code word above the shape>",
                "box_2d": [ , , , ]}}]
}}"""


# The dense fine-text probe joins the suite rather than living as a second
# entry point. Prompt and scorer are IMPORTED from finetext_probe, not copied:
# the probe's assets are committed precisely so the same input scores the same
# everywhere, and a drifted prompt copy would defeat that just as thoroughly as
# a regenerated font would. finetext_probe.py still runs standalone.
from finetext_probe import PROMPT as FINETEXT_PROMPT, score_codes as score_finetext
from finetext_probe import NUM_PREDICT as FINETEXT_NUM_PREDICT, NUM_CTX as FINETEXT_NUM_CTX

# (name, prompt, images, scorer, gen_opts). gen_opts is optional and carries
# per-probe generation overrides; fine-text needs a bigger allowance than the
# suite default (20 codes plus JSON scaffolding do not fit 2200 tokens), and an
# exhausted allowance reads as a vision failure rather than a truncation.
tests = [
    ("scene_single", SCENE_PROMPT.format(w=1920, h=1080), ["scene_hd.png"], score_scene),
    ("document_single", DOC_PROMPT.format(w=1568, h=1568), ["document.png"], score_doc),
    ("multi_3img", MULTI_PROMPT, ["scene_hd.png", "document.png", "chart.png"], score_multi),
    ("bbox_contract", BBOX_CONTRACT_PROMPT.format(w=1920, h=1080), ["scene_hd.png"],
     score_bbox_contract),
    # THE REPRODUCER. Distractor images attached, with an instruction to ignore
    # them. Measured 2026-08-16 under the bbox_type/ref_size schema, this cell is
    # followed in only 5 of 21 runs across the seven-model corpus (3 repeats
    # each) — see docs/maxusai/vision-campaign-2026-08-16-seven-model.md. It is
    # NOT a qwen3.8 quirk, which is how it was first written up:
    #
    #   gemma4 GGUF   0/3  declares xyxy, emits yxyx
    #   gemma4 MLX    2/3  yxyx on one run of three
    #   qwen3.6 GGUF  0/3  declares real [1920,1080], emits norm-1000
    #   qwen3.6 MLX   0/3  declares real [1024,768] — a frame never sent
    #   qwen3.8 GGUF  0/3  declares norm1000, emits real in a ~1.33x frame
    #   qwen3.8 MLX   3/3  —
    #   nemotron GGUF 0/3  declares real, no ref_size, emits norm-1000
    #
    # No GGUF configuration passes, in nine attempts. Every failing cell still
    # LOCATES all six shapes: perfect vision, false self-description — the exact
    # defect a fixed-dialect scorer reports as a bare miss.
    #
    # For qwen3.8 GGUF the single-image probe passes 3/3 and the reasoning
    # variant below passes 3/3, so neither image count nor reasoning load is the
    # trigger on its own. What distinguishes this cell is being told to IGNORE
    # the other images. The mechanism is unexplained; the rates are not. Do not
    # "fix" a failure here by relaxing the scorer.
    ("bbox_contract_multi",
     BBOX_CONTRACT_PROMPT.format(w=1920, h=1080)
     + "\n\nOnly the FIRST image contains the shapes to report; the others are "
       "distractors and must be ignored.",
     ["scene_hd.png", "document.png", "chart.png"], score_bbox_contract),
    ("bbox_contract_reasoning",
     BBOX_CONTRACT_REASONING_PROMPT, ["scene_hd.png", "document.png", "chart.png"],
     score_bbox_contract),
    # Env still wins, as it does for the standalone probe — the override only
    # replaces the suite's default with this probe's, it does not pin it.
    ("finetext", FINETEXT_PROMPT, ["finetext.png"], score_finetext,
     {"num_predict": int(os.environ.get("NUM_PREDICT", FINETEXT_NUM_PREDICT)),
      "num_ctx": int(os.environ.get("NUM_CTX", FINETEXT_NUM_CTX))}),
]

def main():
    global HOST, TAG, MODEL
    HOST = sys.argv[1]
    TAG = sys.argv[2]
    MODEL = sys.argv[3] if len(sys.argv) > 3 else "nemotron3:33b-q4_K_M"
    results = {}
    run_tests = tests
    # ONLY_TESTS takes precedence over the positional [test] arg, but no longer
    # clobbers it — previously the env lookup overwrote argv[4] unconditionally,
    # so the documented positional form was dead.
    only = os.environ.get("ONLY_TESTS") or (sys.argv[4] if len(sys.argv) > 4 else None)
    if only:
        keep = set(only.split(","))
        run_tests = [t for t in run_tests if t[0] in keep]
        missing = keep - {t[0] for t in tests}
        if missing:
            print(f"WARNING: unknown test name(s) ignored: {', '.join(sorted(missing))}")
        if not run_tests:
            print(f"ERROR: no tests matched {only!r}; nothing to run")
            sys.exit(2)
    # NOTE: run_tests is already filtered above. A second per-iteration check
    # comparing `name != only` used to live here, which silently skipped EVERY
    # test whenever ONLY_TESTS held more than one comma-separated name (no single
    # name equals the whole string) — producing an empty scores file that looked
    # like a model failure.
    for entry in run_tests:
        # 5th element is optional per-probe gen overrides; the original 4-tuples
        # keep working unchanged.
        name, prompt, images, scorer = entry[:4]
        gen_opts = entry[4] if len(entry) > 4 else {}
        print(f"--- {name} [{TAG}] ---", flush=True)
        try:
            r = gen(prompt, [b64(i) for i in images], **gen_opts)
        except Exception as e:
            print(f"ERROR: {e}")
            results[name] = {"error": str(e)}
            continue
        text = r.get("response", "")
        open(f"{DIR}/resp_{TAG}_{name}.json", "w").write(text)
        sc = scorer(text)
        sc["prompt_eval_count"] = r.get("prompt_eval_count")
        sc["eval_count"] = r.get("eval_count")
        # Throughput. Ollama reports durations in nanoseconds. Recorded so a run
        # can be compared across backends (Metal vs CPU) as well as scored —
        # additive only, no effect on any existing score field.
        for k in ("total_duration", "load_duration",
                  "prompt_eval_duration", "eval_duration"):
            sc[k] = r.get(k)
        if r.get("eval_duration") and r.get("eval_count"):
            sc["gen_tps"] = round(r["eval_count"] / (r["eval_duration"] / 1e9), 2)
        if r.get("prompt_eval_duration") and r.get("prompt_eval_count"):
            sc["prefill_tps"] = round(
                r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9), 2)
        # The request window these numbers were achieved under. Without it a
        # score is not interpretable: an empty or short result may be the model
        # or may be the cap, and cells measured at different num_ctx are not
        # comparable on throughput (KV size affects decode speed).
        sc["num_ctx"] = default_num_ctx()
        sc["num_predict"] = default_num_predict()
        # Which sampling this cell was measured under. ADR 0005 asks for the
        # runtime configuration to be recorded; without it, a capped cell cannot
        # be attributed to greedy decoding after the fact — which is exactly
        # what happened to the b10353 think-on campaign.
        sc.update(provenance(MODEL, os.environ.get("THINK", "false") == "on"))
        # Record the requested vision budget so a scores file is self-describing:
        # absent means "build default", present means this was a budget-matched
        # control arm. Without this a control run is indistinguishable from a
        # normal one after the fact.
        for env, key in (("IMAGE_MIN_TOKENS", "req_image_min_tokens"),
                         ("IMAGE_MAX_TOKENS", "req_image_max_tokens")):
            if os.environ.get(env):
                sc[key] = int(os.environ[env])
        # num_ctx / num_predict are PER MODEL AND PER TEST — nemotron3's
        # document_single needs 16,421 tokens while its scene_single needs 7,622,
        # and gemma4 terminates inside 10,691 for every test. A run is not
        # interpretable without them: an empty response means "truncated" at one
        # window and "the model would not stop" at another, and the two are
        # indistinguishable after the fact. Reported in the tables as
        # "value (num_ctx)". See ADR 0012.
        sc["req_num_predict"] = r.get("_num_predict")
        sc["req_num_ctx"] = r.get("_num_ctx")
        results[name] = sc
        print(json.dumps(sc, indent=1), flush=True)
    
    open(f"{DIR}/scores_{TAG}.json", "w").write(json.dumps(results, indent=1))
    print("SUITE DONE", TAG)

if __name__ == "__main__":
    main()
