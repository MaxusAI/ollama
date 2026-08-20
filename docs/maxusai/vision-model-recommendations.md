# Vision model recommendations — settings per model, think mode, and use case

**Derived from measurements of 2026-08-20** — the `cudafull1` five-model CUDA
baseline (full CONTEXT ladder; [tasks/cuda-baseline-full-ladder-rerun.md](tasks/cuda-baseline-full-ladder-rerun.md)
holds the verbatim T1/T2 renders and the per-probe thinking-token map) and the
1,120-cell composable bbox factor matrix (`summarize_matrix.py mx` render,
same day). Host `http://10.8.0.6:11497`, build `0.32.14-rc0-dynres-0-ga5d6590`,
GGUF q4_K_M throughout. Every number below traces to those renders; nothing
here is a new measurement.

**Scope caveats.** One host, one build, one quantization, one repeat per cell
— and a think-on cell is one draw, not a constant (ADR 0012 conv. 4: measured
same-cell recall_9px 1/4 vs 2/4 across two draws). Apple/MLX and ROCm are out
of scope (deliberately skipped 2026-08-20). nemotron3 think-on token counts
are understated by an open server bug
([tasks/nemotron-thinkon-evalcount-undercount.md](tasks/nemotron-thinkon-evalcount-undercount.md));
its `s/req` is unaffected.

## A complete request, end to end

Everything below this section explains *why*; this section is the *what*: one
full, working request. It is emitted by
[vision-suite/emit_request.py](vision-suite/emit_request.py), which captures
the payload from `client.py`'s own construction (H9 — one payload builder),
so it cannot drift from what the campaigns actually sent. Reproduce any
variant yourself:

```bash
cd docs/maxusai/vision-suite
python3 emit_request.py gemma4:31b-it-q4_K_M on            # trustable arm, think-on
python3 emit_request.py qwen3.6:35b-a3b-q4_K_M false       # any model / mode
python3 emit_request.py qwen3.8:27b-q4_K_M on bboxm_pin_anc_pos   # any arm
```

**gemma4:31b-it-q4_K_M, think=on, the trustable arm (`bboxm_pin_anc_named`)
— emitted verbatim:**

```
POST http://HOST:11497/api/chat
{
  "model": "gemma4:31b-it-q4_K_M",
  "stream": false,
  "options": {
    "num_predict": 8192,
    "num_ctx": 16384,
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64
  },
  "format": "json",
  "messages": [
    {
      "role": "user",
      "content": "You are a localization service. Find every distinct\ncoloured shape in this image and report where each one is.\n\nUse \"bbox_type\": \"norm1000\" — each axis scaled independently to 0-1000, x by\n1000/width and y by 1000/height. The coordinate space is 1000x1000 whatever the\nimage's shape is.\n\n\nGive each coordinate its own named field: \"x1\", \"y1\", \"x2\", \"y2\". Do not use a\npositional array, and do not declare a \"coord_order\" — named fields state their own order.\n\nDeclare the convention on EVERY object, next to that object's coordinates.\n\nEach box covers the shape itself, not its label text. A box that disagrees with\nits declaration is worse than no answer at all, because a consumer trusts the\ndeclaration.\n\nThe FIRST entry must be a calibration entry with label \"__IMAGE__\" whose\ncoordinates cover the ENTIRE image, corner to corner, in the same convention as\neverything else. Then list the shapes.\n\nRespond with a SINGLE JSON object, no prose:\n{\n  \"objects\": [{\"label\": \"__IMAGE__\", \"bbox_type\": \"norm1000\", \"x1\": , \"y1\": , \"x2\": , \"y2\": },\n              {\"label\": \"<uppercase code word above the shape>\",\n               \"bbox_type\": \"norm1000\", \"x1\": , \"y1\": , \"x2\": , \"y2\": }]\n}",
      "images": [
        "<base64 of visimgs/scene_hd.png>"
      ]
    }
  ]
}
```

Replace the image placeholder with the base64 of your PNG and `HOST` with the
server. Note what the trustable arm actually looks like on the wire: **named
coordinate fields** (`"x1":, "y1":, "x2":, "y2":`) with `"bbox_type":
"norm1000"` declared on every object, and the `__IMAGE__` calibration entry
first. Named fields state their own order, so the whole
`box_2d`/`bbox_2d`/xyxy/yxyx dialect problem simply does not arise — the
dialect arrays only enter with the positional arms
(`emit_request.py <model> on bboxm_pin_anc_pos` offers `box_2d` to gemma and
`bbox_2d` to qwen/nemotron automatically).

**The same request, think=off** (emitted): the prompt and structure are
identical; three things change —

```json
  "think": false,
  "options": {"num_predict": 2200, "num_ctx": 16384, "temperature": 0}
```

(`think: false` appears at top level; when thinking is ON the field is
omitted; card sampling keys drop out.)

**The response schema the prompt asks for**, formatted for readability
(inside the prompt it appears as the escaped one-liner above; `x1…y2` are the
slots the model fills):

```json
{
  "objects": [
    {"label": "__IMAGE__", "bbox_type": "norm1000",
     "x1": 0, "y1": 0, "x2": 1000, "y2": 1000},
    {"label": "<uppercase code word above the shape>", "bbox_type": "norm1000",
     "x1": 0, "y1": 0, "x2": 0, "y2": 0}
  ]
}
```

**And what actually comes back** — gemma4:31b think=on's measured answer to
exactly the request above (`resp_cudafull1_gemma4_31b-it-q4_K_M_thinkon_bboxm_pin_anc_named.json`,
verbatim, six shapes elided to two):

```json
{
  "objects": [
    {"label": "__IMAGE__", "bbox_type": "norm1000",
     "x1": 0, "y1": 0, "x2": 1000, "y2": 1000},
    {"label": "DYNAMO", "bbox_type": "norm1000",
     "x1": 115, "y1": 554, "x2": 252, "y2": 797}
  ]
}
```

The `__IMAGE__` entry at `[0, 0, 1000, 1000]` is the model confirming it
answered in the requested frame; DYNAMO lands within 1–2 norm-1000 units of
ground truth (`[115, 556, 250, 796]`). Strip the calibration entry, convert
each coordinate with `px = v · size / 1000`, done.

**Per-model deltas, think=on** — only `model` and `options` change; the
named-arm prompt is model-independent (emitted for each):

| model | think=on `options` |
|---|---|
| gemma4:31b / 26b-a4b | `{"num_predict": 8192, "num_ctx": 16384, "temperature": 0.0, "top_p": 0.95, "top_k": 64}` |
| qwen3.6:35b-a3b | `{"num_predict": 8192, "num_ctx": 16384, "temperature": 0.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}` |
| qwen3.8:27b, nemotron3:33b | `{"num_predict": 8192, "num_ctx": 16384}` — no card row, packaged defaults apply |

**Handling the response**: check `done_reason` first. `"stop"` → use the
result (strip the `__IMAGE__` entry before consuming the object list, rescale
through it if `bbox_type` was not what you asked for). `"length"` → the
window was too small for the thinking, not a model failure: re-send with the
next rung (`num_ctx` 32768, `num_predict` 24576; then 65536/57344,
131072/122880). The per-model sections below say which rung each model
realistically needs.

## Rules that apply to every model

1. **Sampling**: think-off runs greedy (`temperature: 0` — every baseline
   depends on it). Think-on ALSO runs `temperature: 0` since 2026-08-17
   (`THINK_TEMPERATURE`, fork policy: reproducibility over termination — a
   non-terminating cell under this policy is expected behaviour for the
   affected families, measured by the ladder, not a vision failure), plus
   the card's non-temperature keys where a card is readable (`sampling.py`).
   qwen3.8 and nemotron3 have no readable card row, so they run their
   packaged defaults — qwen3.8's are non-greedy, which is why its think-on
   cells are single draws. Greedy thinking is what produces runaway
   reasoning ([runaway-reasoning-under-think.md](runaway-reasoning-under-think.md));
   this policy accepts that cost knowingly — see `sampling.py`'s header.
2. **Window**: think-off fits everywhere at `num_ctx` 16384 / `num_predict`
   2200. Think-on: never pin a fixed window — start at 16384 with
   `num_predict = num_ctx − 8192` and let the CONTEXT ladder escalate per
   cell (SPEC H4a/H4b). The rung a model needs is a result; the per-model
   sections below list the measured rungs.
3. **Prompt shape**: `pin + anchor + named coords` — state the actual image
   size, ask for an `__IMAGE__` calibration entry, use a named coordinate
   space — is the trustable arm for 7 of 9 measured model×mode groups
   (6.00/6, geometry spread 0.00 for most; factor matrix). The one measured
   exception: **nemotron3 think-on prefers positional over named coords by
   3.78/6**. The anchor is non-negotiable for multi-image bbox work: it flips
   qwen3.8 and nemotron3 ❌→✅ and is the difference between qwen3.6 think-on
   never terminating and finishing in 9,598 thinking tokens.
4. **Coordinate dialect is per model** and must be offered, never assumed:
   `box_2d` for gemma4, `bbox_2d` for qwen3.x and nemotron3
   ([vision-bbox-coordinate-conventions.md](vision-bbox-coordinate-conventions.md)).
   Never trust a model's *declared* coordinate space on adversarial input —
   nemotron3 declares real-pixel and emits norm-1000 (0/6 on the adversarial
   contract cell).
5. **Trust `done_reason`**: `stop` before using any cell as a result;
   `length` means the measurement is unfinished, not that the model failed
   (ADR 0012 conv. 9). For nemotron3 think-on it is the *only* reliable cap
   signal while the eval_count bug is open.
6. **Endpoint**: `/api/chat` with `format: "json"` (the calibrated campaign
   path; `/api/generate` drops reasoning text).

## The exact request JSON

Everything above lands on the wire through one payload shape (`client.py` is
the single request path, SPEC H9). The campaign's exact think-on request for
gemma4:

```json
{
  "model": "gemma4:31b-it-q4_K_M",
  "stream": false,
  "format": "json",
  "messages": [{"role": "user", "content": "<prompt>", "images": ["<base64>", "..."]}],
  "options": {
    "num_predict": 8192,
    "num_ctx": 16384,
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64
  }
}
```

Think-off differs in three places: `"think": false` appears at top level,
`options.num_predict` is `2200`, and the card keys drop out (options carry
only `num_predict`, `num_ctx`, `temperature: 0`).

| key | values used | effect |
|---|---|---|
| `stream: false` | always | one JSON body back; `done_reason`, `eval_count`, `thinking`, `response` arrive together |
| `format: "json"` | always | grammar-constrained output. On thinking models this engages the server's deferred-thinking / constrained-continuation machinery — the path where nemotron3's `eval_count` undercount lives |
| `think` | `false` sent when thinking is off; **omitted** when on | tri-state on purpose: the template renders differently depending on the field's presence, worth a token or two. Pin it in both directions only when an experiment requires it (`send_think=True` in `client.py`) |
| `messages[].images` | base64 strings | `/api/chat` form; `/api/generate` uses top-level `prompt` + `images`. An empty image list must OMIT the key — `"images": []` is neither an image request nor a text-only one |
| `options.num_predict` | 2200 off / `num_ctx − 8192` on | the hard generation cap. Hitting it returns `done_reason: "length"` — an unfinished measurement (capped), never a score. Think-on derives it from the rung so the pair stays coherent as the ladder climbs |
| `options.num_ctx` | 16384 start; ladder 32768 → 65536 → 131072 | the shared window. The server hard-rejects (400) any request where prompt + `num_predict` exceeds it — no silent truncation. KV doubles per rung (decode speed drops), and termination itself is window-dependent: nemotron3 fine-text capped a 24,576 budget at 32768 yet finished in 4,909 tokens at 65536 |
| `options.temperature` | `0`, both modes | fork policy (`THINK_TEMPERATURE`). Reproducible cells; the cost is non-termination pressure on qwen3.6/gemma-26b-a4b free-form think-on, handled by the ladder + `done_reason`, not by raising temperature |
| `options.top_p / top_k / min_p / presence_penalty` | gemma4: `0.95 / 64`; qwen3.6: `0.95 / 20 / 0.0 / 1.5`; qwen3.8, nemotron3: **not sent** | card-sourced, think-on only, never tuned by us. With temperature 0 they are mostly inert; absent rows mean packaged defaults apply (qwen3.8's are non-greedy — its think-on cells differ run to run) |
| `options.kv_cache_type` | only when `KV_CACHE_TYPE` env set | runner option, reloads the model; raise for uncapped think-mode probes |
| `options.image_min_tokens / image_max_tokens` | only when env set | fork-only per-request vision budget, arch-gated to gemma4 and nemotron_h_omni. Pin to upstream defaults to build a budget-matched control against a stock server |
| `raw: true` | token-counting only | `/api/generate` only: skips the chat template so `prompt_eval_count` counts bare text — `token_split.py --server`'s mechanism. Never used for scored cells |

Response keys the recommendations gate on: **`done_reason`** (`stop` =
finished, `length` = capped/unfinished, absent = no verdict — dropped
connection or pre-2026-08-20 cell), **`eval_count`** (all generated tokens,
thinking included; understated for nemotron3 think-on while its bug is
open), **`prompt_eval_count`**, and the separated **`thinking`** /
**`response`** texts (only `/api/chat` returns both — `/api/generate` drops
reasoning text).

### Response-schema vocabulary: dialects, spaces, order, anchor, pin

The trustable arm sidesteps most of this with named fields — see the
complete request at the top of this document. This vocabulary matters in two
situations: when a prompt uses positional arrays (the `*_pos` arms), and
whenever you consume a model's free-form output, because models volunteer
these notations unprompted.

One real object in every notation — DYNAMO, ground truth
`[220, 600, 480, 860]` pixels in a 1920×1080 image
([vision-bbox-coordinate-conventions.md](vision-bbox-coordinate-conventions.md)
is the full reference):

```json
{"label": "DYNAMO", "bbox_2d": [220, 600, 480, 860]}   // real px,  xyxy  (qwen/nemotron field name)
{"label": "DYNAMO", "bbox_2d": [115, 556, 250, 796]}   // norm1000, xyxy  (x·1000/W, y·1000/H)
{"label": "DYNAMO", "box_2d":  [556, 115, 796, 250]}   // norm1000, yxyx  (gemma/Gemini documented order)
{"label": "DYNAMO", "box_2d":  [115, 556, 250, 796]}   // what gemma4 MEASURABLY emits: norm1000, xyxy
```

- **`box_2d` vs `bbox_2d`** — pure field-name dialect: gemma4/Gemini are
  trained toward `box_2d`, qwen-vl and nemotron toward `bbox_2d`. Offer the
  model its own name (demanding one measures naming compliance, not vision);
  a consumer should accept `bbox`, `bbox_2d`, `box_2d` alike, as the suite's
  scorers do.
- **`norm1000`** — each axis independently scaled to 0–1000:
  `x_norm = x_px · 1000 / W`. Convert back with `x_px = x_norm · W / 1000`.
  Also seen: `norm1` (0.0–1.0) and `real` (absolute pixels — which is only
  meaningful together with a frame, see `ref_size`).
- **`xyxy` vs `yxyx`** — `[x1, y1, x2, y2]` (qwen convention) vs
  `[y1, x1, y2, x2]` (the documented gemma/Gemini `box_2d` order). Measured
  in this corpus gemma4 GGUF actually emits xyxy while declaring it; a
  robust consumer tries both orders, as `score_multi` does.
- **The self-describing contract** (ms-swift vocabulary, so a conforming
  response is directly usable as fine-tuning data):

  ```json
  {
    "bbox_type": "real",
    "ref_size": [2500, 1406],
    "coord_order": "xyxy",
    "objects": [{"label": "DYNAMO", "box_2d": [286, 781, 625, 1119]}]
  }
  ```

  `ref_size` is mandatory with `real`: "absolute pixels" is a convention
  *plus a frame*, and qwen3.8 GGUF demonstrably reports in a ~1.3× frame —
  honestly declaring `ref_size [2500, 1406]` for a 1920×1080 input. Divide it
  out and its IoU goes 0.079 → 0.909. **Never trust the declaration alone on
  adversarial input**: nemotron3 declares `real` and emits norm-1000 (0/6 on
  that cell) — the direction that silently halves every box.

- **Anchored** — ask that the objects array *begin* with a calibration entry
  covering the entire image in the same convention as every other box
  (prompt wording, verbatim: *"whose `label` is `__IMAGE__` and whose `bbox`
  covers the ENTIRE image, corner to corner, in exactly the same convention
  as every other box … If you resized image 1 internally, use the size YOU
  used"*). The model then declares its true frame in-band:

  ```json
  "key_objects": [
    {"label": "__IMAGE__", "bbox_2d": [0, 0, 2560, 1440]},
    {"label": "DYNAMO",   "bbox_2d": [293, 800, 640, 1147]}
  ]
  ```

  (that `[0, 0, 2560, 1440]` is qwen3.8's measured self-declaration for a
  1920×1080 input). Effect: the consumer rescales through the declared frame
  instead of guessing — measured to flip qwen3.8 and nemotron3 multi q4
  ❌→✅, and to turn qwen3.6 think-on from never-terminating into a
  9,598-token clean finish. Strip the `__IMAGE__` entry before treating the
  list as content.

- **Pinned** — prompt-side, no JSON key: the prompt states the frame and the
  space outright (verbatim: *"Bounding boxes use `norm1000` … formatted
  `[x1, y1, x2, y2]` … The image is exactly {w} pixels wide and {h} pixels
  tall, but report NORMALIZED 0-1000 values"*). Effect: removes the model's
  need to name a frame at all. The factor matrix measured pin as the largest
  single marginal for the weaker configurations (gemma-26b-a4b think-off
  +1.79/6, nemotron3 think-off +1.54/6) — and `pin + anchor + named` together
  are the trustable arm. The one caution transfers from the qwen3.6 cell:
  a pinned frame must state the size actually sent — a prompt that lies
  about its own input measures obedience to a false premise.

## Per-model settings

### gemma4:26b-a4b-it-q4_K_M (MoE)

| | think=false (recommended) | think=on |
|---|---|---|
| window | 16384 / 2200 | needs **65536** for scene (ladder-measured) |
| quality | scene IoU 0.973, doc bbox 0.756, fine text 4/4/4/3/3, multi ✅✅✅ | scene **collapses to 0.334, boxes 0/6**; fine text 4/4/4/4/3 |
| cost | 150 tok/s, 5.0 s/req, 725 req/h | 7,153 thinking tok on scene; 62 req/h (12× slower) |

**Use think-off.** Best quality-per-second in the fleet. If thinking is
required, use the full pin+anchor+named prompt shape (its matrix arm is
6.00/6 in both modes — the collapse is specific to free-form scene prompts)
and budget the 65536 window. Dialect `box_2d`.

### gemma4:31b-it-q4_K_M (dense)

| | think=false | think=on |
|---|---|---|
| window | 16384 / 2200 | **16384 suffices — never escalated** |
| quality | scene 0.962, fine text 4/4/4/4/3 (best small-tier OCR), multi ✅✅✅ | scene 0.962 (no think damage), fine text 4/4/4/4/3, multi ✅✅✅ |
| cost | 56 tok/s, 296 req/h | 833 thinking tok on scene; 102 req/h |

**The safe-either-mode model.** Only model with zero think-on degradation,
zero ladder escalation, and matrix 6.00/0.00 in both modes. Slowest decode —
choose it for correctness-critical work, small-glyph OCR (9px 4/4, 7px 3/4),
or wherever think-on is mandated. Dialect `box_2d`.

### qwen3.8:27b-q4_K_M (dense)

| | think=false | think=on |
|---|---|---|
| window | 16384 / 2200 | 16384 suffices |
| quality | scene **0.977 (best)**, doc bbox 0.550, multi q4 ❌ unanchored / ✅ anchored | scene 0.975, doc bbox **0.858 (best)**, multi q4 ❌ unanchored / ✅ anchored |
| cost | 65 tok/s, 366 req/h | frugal thinker: 431–1,421 tok per probe, 21,750 suite total; 188 req/h |

**Always anchor.** It grounds near-perfectly but in an internally rescaled
frame (measured ~1.22×, self-reported in its own thinking stream); without a
calibration entry any cross-frame consumer of its boxes scores it wrong.
**Avoid <9px text**: its small-tier misses are sub-glyph optical confusions
(M↔N, W↔K, digits 1↔3↔5↔9) in structurally correct codes — mode-independent,
so thinking cannot fix it. Best pick for document bbox work under thinking.
Dialect `bbox_2d`.

### nemotron3:33b-q4_K_M

| | think=false (recommended) | think=on |
|---|---|---|
| window | 16384 / 2200 | **32768 for multi; fine text needed the 65536 window** (terminated in 4,909 tok there after capping 24,576 at 32768 — window-dependent termination) |
| quality | scene 0.870, invoice fields 5/5 ✅✅, **doc bbox 0.044 — unusable** | scene 0.577, doc bbox 0.114, multi ✅✅❌ / anchored ✅✅✅ |
| cost | **209 tok/s gen, 4,797 prefill, 3.0 s/req, 1,196 req/h — fleet's fastest** | 148 req/h; token counts understated (open bug) |

**The throughput king for field extraction** — invoice items/totals/serials
all correct at 4× the req/h of anything else. **Never consume its document
bboxes** (0.044/0.114 both modes). Under thinking: use **positional coords**
(named costs it −3.78/6), anchor everything, expect the 32768 rung, and gate
on `done_reason`, not token counts. Dialect `bbox_2d`, but verify via anchor
— it misdeclares its space under adversarial prompts. 7px OCR: 0/4.

### qwen3.6:35b-a3b-q4_K_M (MoE)

| | think=false (recommended) | think=on |
|---|---|---|
| window | 16384 / 2200 | ≥32768; **three probes never terminate even at 131072/122,880** (`multi_3img` unanchored, two contract probes) |
| quality | scene 0.975, multi ✅✅✅, fine text 4/4/4/2/2 | scene 0.717 @32768; grounds 6.00/6 in *every* terminating matrix cell |
| cost | 95 tok/s, 545 req/h | **275,791 thinking tok** across the uncapped suite (12.7× qwen3.8); 15 req/h |

**Think-off by default.** Think-on is a termination lottery on free-form
prompts: when it stops, it is perfect; whether it stops depends on the prompt
shape. If think-on is required: anchored + pinned prompts only (anchored
multi finishes in 9,598 tok ✅✅✅; unanchored burns 122,880 and never stops),
window ≥32768, and treat any unanchored bbox request as a non-termination
risk. Dialect `bbox_2d`.

## By scenario

| scenario | pick | mode | key settings |
|---|---|---|---|
| Scene/object grounding, throughput matters | gemma4:26b-a4b | off | 16384/2200, `box_2d`, greedy |
| Scene grounding, max accuracy | qwen3.8 | off | 16384/2200, **anchored**, `bbox_2d` |
| Invoice/field extraction at scale (no boxes) | nemotron3 | off | 16384/2200; 1,196 req/h; ignore its bboxes |
| Document extraction **with** name bboxes | qwen3.8 | on | 16384, anchored, `bbox_2d` (0.858) |
| Small-print OCR (≤9px glyphs) | gemma4:31b | either | 16384/2200; the only model ≥3/4 at 7px |
| Multi-image cross-referencing | gemma4:31b or 26b-a4b | either | ✅✅✅ even unanchored; anchor anyway |
| Reasoning-mandated vision tasks | gemma4:31b | on | 16384 suffices, no think damage, cheap thinking (≤5,393 tok worst probe) |
| Reasoning + tight token budget | qwen3.8 | on | 16384, anchored; frugalest thinker |

## Overall

**gemma4:31b-it-q4_K_M is the overall recommendation** when one model must
cover everything: the only one with no think-mode degradation, no window
escalation, best small-glyph OCR, clean multi-image sweeps in both modes, and
matrix-perfect grounding under the pin+anchor+named shape — its price is
decode speed (56 tok/s). When throughput matters more than 7px text and
thinking is off, **gemma4:26b-a4b** delivers near-identical quality at 2.5×
the req/h. Split fleets pair **nemotron3 think-off** for high-volume field
extraction with **qwen3.8 anchored** for anything that consumes boxes.

And the single highest-leverage setting across every model and mode is not a
model choice at all: **ask for the `__IMAGE__` calibration anchor**. Measured
today, it turned two ❌ multi cells into ✅ sweeps, distinguished frame errors
from grounding failures, and converted qwen3.6's think-on non-termination
into a 9,598-token clean finish.
