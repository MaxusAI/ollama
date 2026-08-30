# Vision campaign 2026-08-31 — the nvfp4 fleet on the 0.33.2 MLX runner

One host, one build. Apple Silicon `10.8.0.3`, native serve on `:11436`
(`serve-apple-mlx.sh`, cold server per cell, `OLLAMA_MAX_LOADED_MODELS=1`),
build **`0.33.2-maxusai-2b95b4a5`** — llama.cpp `b10630`, MLX `c793734e`, with
upstream's grammar engine in place of the fork's constrained sampling
([ADR 0033](adr/0033-mlx-constrained-sampling-adopts-upstreams-engine.md)).
Campaign tag `mlx0332nv1`. **Every cell ran at `powermode=2`**, which matters
below.

This repeats [the 2026-08-28 campaign](vision-campaign-2026-08-28-mlx0330-nvfp4.md)
on the next fold, so the two are directly comparable: same host, same suite,
same sampling, same power mode on every cell but one (noted in Limits).

**The question.** Does the v0.33.2 fold — a payload move on both pins plus a
different structured-output engine — change what these models produce, or what
they cost? Preflight already said the plumbing is sound
(`mlx-metal-0-33-2`, VERDICT PASS); this measures output and throughput.

**Scope was operator-directed.** Think-off runs all five nvfp4 tags. Think-on
runs **only `gemma4:31b-nvfp4` and `qwen3.8:27b-nvfp4`** — the two cells that
converged at the first rung on 0.33.0. The other three are deliberately absent
and the reason differs per model; see §3. The repo convention is to always run
both think modes, so this narrowing is a cost decision stated here rather than
left for a reader to infer from a short table.

## The request this campaign sends

`emit_request.py gemma4:31b-nvfp4 false` — pasted verbatim (SPEC H7/H9; the
campaign drove `http://127.0.0.1:11436`, the `HOST:11497` placeholder below is
the emitter's):

```
POST http://HOST:11497/api/chat
{
  "model": "gemma4:31b-nvfp4",
  "stream": false,
  "options": {
    "num_predict": 2200,
    "num_ctx": 16384,
    "temperature": 0
  },
  "format": "json",
  "think": false,
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

## 1. Think-off is unchanged, on every model

`summarize_engine_compare.py --think false --prefix mlx0332nv1_`, verbatim:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.954** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.969** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.959** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.964** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 4 | 3 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 34 | 1984 | 16.5 | 218 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 541 | 54 | 8502 | 10.2 | 353 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 16 | 1083 | 34.7 | 104 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 25 | 1940 | 23.2 | 155 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 77 | 14251 | 7.4 | 490 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.33.2-maxusai-2b95b4a5 · think=false

**Quality is the same build to build.** Four of five scene IoUs reproduce the
0.33.0 value to three decimals (0.954 / 0.969 / 0.959 / 0.999); qwen3.6 moves
0.963 → 0.964, one digit in the third decimal. Every cell converts 6/6 boxes,
labels and colours, finds the serial, and extracts 5/5 invoice items. Fine-text
is identical except qwen3.6's 7px tier, 1 → 2 of 4.

No cell escalated: all five converged at the 16384 start rung, as on 0.33.0.
So the fold — new MLX pin, new grammar engine — is **inert for vision quality**.

## 2. Think-on, for the two cells that terminate

`summarize_engine_compare.py --think on --prefix mlx0332nv1_`, verbatim:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.958** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **1.000** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2446 | 32 | 259 | 81.8 | 44 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 1 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1181 | 34 | 1723 | 36.1 | 100 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.33.2-maxusai-2b95b4a5 · think=on

Quality holds here too: 31b 0.959 → 0.958, qwen3.8 0.990 → 1.000, both 6/6 and
5/5 throughout. Two differences worth recording rather than smoothing over:

- **31b needed one escalation.** `bboxm_free_noanc_pos` capped at 16384 and
  converged at 32768, where on 0.33.0 the cell was 27/27 at the first rung. The
  ladder did its job and the published row is the settled rung, but the rung
  path is not stable across builds.
- **qwen3.8 lost `multi_3img` q4-bbox** (❌ where 0.33.0 had ✅) while the
  anchored variant still passes. That is the anchored/unanchored split
  [2026-08-19 documented](vision-learnings-log.md), not a grounding failure —
  and it is exactly why the two are separate columns.

## 3. gemma4:26b-nvfp4 decodes 38% slower, and nothing else does

The one real regression in this campaign. Median `gen_tps` across all 27
think-off arms per cell:

| Model | 0.33.2 median | 0.33.0 median | Δ | 0.33.2 range | 0.33.0 range | eval_count total |
|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | 33.1 | 32.9 | **+1%** | 25.1–35.1 | 20.9–53.9 | same |
| gemma4:26b-nvfp4 | 54.0 | 87.4 | **-38%** | 50.4–76.5 | 75.3–95.6 | same |
| gemma4:31b-nvfp4 | 16.5 | 14.0 | **+18%** | 14.9–18.8 | 12.1–16.1 | same |
| qwen3.8:27b-nvfp4 | 24.2 | 21.2 | **+14%** | 22.7–25.4 | 18.9–21.8 | 15366 vs 15383 |
| qwen3.6:35b-a3b-nvfp4 | 90.8 | 78.4 | **+16%** | 66.0–113.0 | 72.0–104.3 | 14092 vs 14241 |

**The 26b number is not an artefact.** The distributions barely overlap
(0.33.0 bottoms out at 75.3, 0.33.2 tops out at 76.5, across 27 arms each), the
**total tokens generated are identical** — 13,260 both runs, so this is purely
time per token and not a different amount of work — and the cell's wall clock
went 3m29s → 5m11s, a +49% that agrees with the per-arm figure. Prefill also
dropped, 9,284 → 8,502 tok/s.

**It is not thermal or system-wide**, because every other model got *faster* on
the same build in the same session: 31b +18%, qwen3.6 +16%, qwen3.8 +14%.

**Both cells ran at `powermode=2`**, verified from the two campaign logs rather
than the score blocks — the 0.33.0 26b block carries `powermode: None`, because
the stamp landed later than that cell. The log line is the authoritative record
and reads `powermode=2` for every 0.33.0 think-off cell except 12b.

**Not investigated** (operator decision, 2026-08-31). Recorded here so the next
person does not rediscover it, and so a 26b throughput number from this build is
not compared against a 0.33.0 one without knowing.

## 4. Limits

- **Think-on covers two models of five, by design.** `gemma4:12b-nvfp4` is
  declared in `DESCOPED_CELLS` — it does not terminate and the ladder cannot fix
  it. `gemma4:26b-nvfp4` and `qwen3.6:35b-a3b-nvfp4` are standing at the 131072
  ceiling from the previous campaign, so the driver skips them. Neither absence
  means the model was untested; both mean the measurement was priced and
  declined.
- **The 12b throughput comparison crosses a power-mode boundary.** Its 0.33.0
  think-off cell ran at `powermode=0` and this one at `2`, so the +1% in §3 is
  the one row in that table that is not a like-for-like reading. Every other row
  is 2 → 2.
- **No gate catches a throughput regression.** Preflight asserts token
  accounting and payload identity, not speed; §3 was found by comparing two
  campaigns by hand. A 38% decode regression currently ships silently.
- **n=1 per cell.** Quality reproduces well enough here to be convincing, but
  nothing in this campaign measures run-to-run spread.
- **Scores are on-host, untracked**
  (`docs/maxusai/vision-suite/scores_mlx0332nv1_*.json`, seven files), per the
  campaign convention; this document is the durable record.
