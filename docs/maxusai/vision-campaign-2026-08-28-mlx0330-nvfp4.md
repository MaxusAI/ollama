# Vision campaign 2026-08-28 — the nvfp4 fleet on the 0.33.0 MLX runner

One host, one build. Apple Silicon `10.8.0.3`, native serve on `:11436`
(`serve-apple-mlx.sh`, cold server per cell, `OLLAMA_MAX_LOADED_MODELS=1`),
build **`0.33.0-maxusai-21cfe88e`** — llama.cpp `b10488`, MLX `27fec909` —
the first vsuite campaign on the v0.33.0 fold ([PR #217](https://github.com/MaxusAI/ollama/pull/217),
Metal half closed in [PR #225](https://github.com/MaxusAI/ollama/pull/225)).
Campaign tag `mlx0330nv1`. Scope was operator-directed: **nvfp4 tags only** —
no GGUF, no mxfp8, no mlx-bf16 — so no engine comparison is possible from
this data; every row is MLX.

Sampling per cell is recorded in the scores: `card:gemma4+temp0` /
`card:qwen*+temp0` (temperature 0 with the model card's top_p/top_k —
[runaway-reasoning-under-think.md](runaway-reasoning-under-think.md)'s fix).
Driver: `run_engine_compare.sh` with the CONTEXT ladder to 131072.

**The question.** Does the 0.33.0 build serve the nvfp4 fleet correctly, and
what does each model cost? Asked immediately after the fold's preflight
(`mlx-metal-0-33-0`, VERDICT PASS) — preflight proves plumbing and token
accounting; this campaign measures output quality and throughput.

## The request this campaign sends

`emit_request.py gemma4:31b-nvfp4 on` — pasted verbatim (SPEC H7/H9; the
campaign drove `http://127.0.0.1:11436`, the `HOST:11497` placeholder below is
the emitter's):

```
POST http://HOST:11497/api/chat
{
  "model": "gemma4:31b-nvfp4",
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

## 1. The fleet is fully functional think-off — all five converge at 16384

`summarize_engine_compare.py --think false --prefix mlx0330nv1_`, verbatim:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.954** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.969** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.959** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.963** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 4 | 3 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 42 | 2929 | 13.2 | 273 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 541 | 90 | 9744 | 6.2 | 581 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 13 | 752 | 44.8 | 80 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 21 | 1622 | 27.2 | 132 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 1 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 543 | 75 | 13926 | 7.4 | 488 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.33.0-maxusai-21cfe88e · think=false

No cell escalated: every think-off cell converged at the 16384 start rung
(consistent with [ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) —
think-off is the vision-work configuration). Quality readings: 31b is the
strongest fine-text reader (4/4 down to 9px); qwen3.8 has the best scene IoU
(0.999) and the only 5/5 `name_bbox` in-band. Throughput tracks architecture:
the active-subset tags (`26b`, `35b-a3b`) decode at 75–90 tok/s and 488–581
req/h, dense `31b` at 13 tok/s / 80 req/h.

## 2. Think-on splits the gemma4 family: 12b runs away, 31b converges at rung 1

`summarize_engine_compare.py --think on --prefix mlx0330nv1_`, verbatim:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.959** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2401 | 27 | 230 | 97.1 | 37 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.33.0-maxusai-21cfe88e · think=on

**gemma4:31b-nvfp4 think-on converged at the first rung**: 27/27 arms,
`eval_count` max **3,539** against the 8,192 budget, median 1,457. Quality is
identical to its think-off row (IoU 0.959, same extraction and fine-text
scores) at 2.4× the response cost (37 vs 80 req/h).

**gemma4:12b-nvfp4 think-on never converged.** Under the identical sanctioned
sampling (`card:gemma4+temp0`), 25/27 arms capped at 8,192; after climbing the
ladder to 65536 (`num_predict` 57,344), 20/27 — every `bboxm_*` and nearly
every `bbox_contract_*` arm — were still capped. The 131072 ceiling attempt
was abandoned mid-rung by operator descope after ~9.5 h wall for one cell.
The full record, scoped so it does not overclaim against the 2026-08-13
sampling finding, is the
[2026-08-28 learnings-log entry](vision-learnings-log.md#2026-08-28--model-behaviour);
the seven arms that did finish (`scene_single_pinned`, `document_single`,
`multi_3img`, `multi_3img_anchored`, `bboxm_pin_noanc_named`,
`bbox_contract_anchored`, `finetext`) are consistent with the anchored/pinned
arms bounding reasoning, as
[2026-08-19 found for qwen3.6](vision-campaign-2026-08-19-qwen36-anchored.md).

**Its cell is OPEN, not failed.** `scores_mlx0330nv1_gemma4_12b-nvfp4_thinkon.json`
holds 7 finished rows plus 20 capped rows whose highest recorded rung is 65536
(ADR 0012 conv 9: a capped row is an unfinished measurement). No think-on
table row is rendered for it, and none may be assembled from that file.

## 3. Limits

- **No engine axis.** nvfp4-only scope means MLX-only; nothing here compares
  MLX to GGUF on this build.
- **Power-mode boundary.** The 12b think-off cell ran at `powermode=0`
  (2026-08-27 22:01); every other cell ran at `powermode=2`. Throughput
  comparisons against the 12b think-off row cross that boundary; all other
  rows are internally comparable. Stamped per cell in the campaign log.
- **think-on n=1 per family member.** The 12b/31b termination split is one
  cell each; the 26b and qwen think-on cells were descoped and remain
  unmeasured on this build.
- **Scores are on-host, untracked** (`docs/maxusai/vision-suite/scores_mlx0330nv1_*.json`,
  seven files), per the campaign convention; this document and the learnings
  entry are the durable record.
