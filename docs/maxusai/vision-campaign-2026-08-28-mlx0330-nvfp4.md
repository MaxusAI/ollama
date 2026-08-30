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

## 2. Think-on does not split by family or by size — it splits by termination

All five cells are now measured (the 26b and qwen cells ran overnight
2026-08-29 → 2026-08-30). The outcome is binary per model and does not track
family, parameter count, or the active-subset/dense axis: **two converge at the
first rung, one converges only after climbing to the ceiling, and two never
terminate at any rung the ladder offers.**

| Cell | Converged arms | Highest rung used | Verdict |
|---|---|---|---|
| gemma4:31b-nvfp4 | 27/27 | 16384 | converged at rung 1 |
| qwen3.8:27b-nvfp4 | 27/27 | 16384 | converged at rung 1 |
| gemma4:26b-nvfp4 | 26/27 | 131072 | one arm standing at the ceiling |
| qwen3.6:35b-a3b-nvfp4 | 18/27 | 131072 | nine arms standing at the ceiling |
| gemma4:12b-nvfp4 | 7/27 | 65536 | DESCOPED 2026-08-30, ceiling never attempted |

`summarize_engine_compare.py --think on --prefix mlx0330nv1_`, verbatim:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:26b-nvfp4 | **MLX** | 16384/131072 ⚠ | **0.972** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.990** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/131072 ⚠ | capped | capped | capped | 5/5 · 5/5 · ✅ | 3 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.959** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:26b-nvfp4 | **MLX** | 16384/131072 ⚠ | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | capped | — | 3644 | 84 | 877 | 45.4 | 79 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 849 | 27 | 983 | 34.7 | 104 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384/131072 ⚠ | 4 | 4 | 4 | 2 | 3 | capped | capped | capped | ≥122880 ⚠ | 70 | 11158 | capped | capped |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2401 | 27 | 230 | 97.1 | 37 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.33.0-maxusai-21cfe88e · think=on

**gemma4:31b-nvfp4 and qwen3.8:27b-nvfp4 converged at the first rung.** 27/27
arms each; `eval_count` max **3,539** (median 1,457) and **3,188** (median
1,341) against the 8,192 budget. For both, quality is unchanged from their
think-off rows — 31b holds IoU 0.959 and its 4/4-to-9px fine-text reading, and
qwen3.8 moves 0.999 → 0.990 with identical box/label/colour and extraction
scores — at 2.2–2.4× the response cost (37 vs 80, 104 vs 132 req/h). This
campaign is n=1 per cell, so that IoU delta is not resolvable against
run-to-run spread; what it does rule out is a think-on *improvement* worth the
cost. On this evidence think-on buys
nothing for vision work on either model, which is
[ADR 0022](adr/0022-thinking-is-off-for-vision-work.md)'s position.

**gemma4:26b-nvfp4 needed the whole ladder for one arm.** 26/27 converged, but
the cell climbed 16384 → 32768 → 65536 → 131072 because `multi_3img_anchored`
capped at every rung, ending at the ceiling with 122,880 thinking tokens and
**zero answer characters**. The other 26 arms finished with max `eval_count`
6,869 (median 3,539) — so a single arm cost this cell roughly 4½ hours and
three rung escalations, and its table cells read `capped` rather than a score.

**qwen3.6:35b-a3b-nvfp4 is the second non-terminating cell, and the worst one.**
18/27 converged (rungs 16384 and 32768, max `eval_count` 22,823 — already an
order of magnitude above the gemma4 cells), and **nine arms never terminate**:
`bboxm_pin_anc_pos`, `bboxm_free_anc_named`, `bboxm_free_noanc_named`,
`scene_single_anchored`, `scene_single`, `multi_3img`, `multi_3img_anchored`,
`bbox_contract_reasoning`, `bbox_contract_real_1img`. All nine capped at 65536
and then again at the 131072 ceiling, every one producing 122,880 thinking
tokens and **zero answer characters** at ~70 tok/s — about 30 minutes per arm.

The ladder bought nothing here, and that is the finding: doubling the window
doubled the thinking and changed no outcome. Both rungs are total
non-termination, not a budget that was nearly enough. The cell cost ~8 hours
wall (16:29 → 00:34) for nine unscored arms, and because `scene_single*` and
`multi_3img*` are among them, the scene-grounding and multi-image columns for
this model cannot be filled from this campaign at all.

**Both standing cells are marked, and future campaigns will skip them.** The
driver stamped `"ladder_not_converged_at": 131072` into each still-capped arm's
block — 1 arm in `scores_mlx0330nv1_gemma4_26b-nvfp4_thinkon.json`, 9 in
`scores_mlx0330nv1_qwen3_6_35b-a3b-nvfp4_thinkon.json`, and none anywhere else.
`ceiling-standing <scores> 131072` accordingly exits 0 (standing → cell
skipped, zero restarts and zero inference) for both, and 1 for the converged
cells. This is the ceiling machinery from [PR #233](https://github.com/MaxusAI/ollama/pull/233)
doing its job on real campaign data for the first time; raising `CTX_MAX` above
131072 is what reopens a standing cell.

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

**Its cell is DESCOPED, not failed — and no longer open.**
`scores_mlx0330nv1_gemma4_12b-nvfp4_thinkon.json` holds 7 finished rows plus 20
capped rows whose highest recorded rung is 65536 (ADR 0012 conv 9: a capped row
is an unfinished measurement). No think-on table row is rendered for it, and
none may be assembled from that file.

As of 2026-08-30 the cell is closed by operator decision rather than left open:
`(gemma4:12b-nvfp4, "on")` is declared in `DESCOPED_CELLS`
(`summarize_engine_compare.py`), so `run_engine_compare.sh` skips it at every
rung and a render prints `⚠ DESCOPED` in place of a row instead of counting its
absent arms as an incomplete campaign. The 131072 ceiling will not be attempted.
What that measurement would have bought is already available cheaper: 31b
converges at rung 1 with quality identical to its think-off row, and 26b needs
one arm's worth of escalation. **think-off for this model is unaffected** and
still renders in §1 above.

## 3. Limits

- **No engine axis.** nvfp4-only scope means MLX-only; nothing here compares
  MLX to GGUF on this build.
- **Power-mode boundary.** The 12b think-off cell ran at `powermode=0`
  (2026-08-27 22:01); every other cell ran at `powermode=2`. Throughput
  comparisons against the 12b think-off row cross that boundary; all other
  rows are internally comparable. Stamped per cell in the campaign log.
- **think-on n=1 per cell.** Every termination verdict here is a single
  observation per model. n=1 is enough to establish that a cell *can* fail to
  terminate (nine arms × two rungs is not noise), and not enough to rank the
  converging cells against each other on quality.
- **Two cells carry no scene-grounding or multi-image think-on data at all.**
  qwen3.6's `scene_single*` and `multi_3img*` arms are among the nine standing
  at the ceiling, and 26b's `multi_3img_anchored` likewise; those table cells
  read `capped`, which is an absent measurement, not a zero. Nothing may be
  averaged or ranked across them (ADR 0012 conv 9).
- **`num_predict` was not a hard bound in one arm.** qwen3.6 think-on
  `bbox_contract_pinned` reports `eval_count` 8,244 against `num_predict`
  8,192 — 52 tokens over — with `done_reason: "stop"` and a real 649-character
  answer. It is correctly treated as converged (SPEC H5 is done_reason-first,
  which is exactly why it survives), but it is the only arm in ~270 across this
  campaign where the reported token count exceeds its own budget, and the
  mechanism is unexplained. Worth noting wherever `num_predict` is assumed to
  bound `eval_count`.
- **Scores are on-host, untracked** (`docs/maxusai/vision-suite/scores_mlx0330nv1_*.json`,
  ten files), per the campaign convention; this document and the learnings
  entry are the durable record.
