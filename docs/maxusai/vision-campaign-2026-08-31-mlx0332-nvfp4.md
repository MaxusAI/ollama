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

## 3. Throughput on this host is not reproducible enough to compare across campaigns

**Read this before quoting any tok/s number in this document, or in the
2026-08-28 one.**

The first draft of this section reported a 38% decode regression on
`gemma4:26b-nvfp4`: median `gen_tps` across 27 think-off arms fell 87.4 → 54.0
against the 0.33.0 campaign, with what looked like strong support — the two
distributions barely overlapped (0.33.0 bottoming at 75.3, 0.33.2 topping at
76.5), the total tokens generated were identical at 13,260 so it was
time-per-token rather than more work, the cell's wall clock agreed at +49%, and
every other model in the same session got *faster*, which appeared to rule out
thermal or system-wide causes.

**It was wrong.** Re-measuring the same four arms on the same binary, same
session, same `powermode=2`, three times in a row:

| run | median gen_tps | per-arm |
|---|---|---|
| 1 | 105.9 | 107 110 101 104 |
| 2 | 100.4 | 108 104 97 95 |
| 3 | 90.5 | 101 99 82 79 |
| — | — | |
| the campaign's 26b cell | **53.8** | 54 54 51 58 |
| 0.33.0's 26b cell | 88.6 | 90 92 86 87 |

The campaign figure is **1.7x below the slowest repeat** of the identical
configuration. So 53.8 is not what this build does; it is what that cell did
that night. Both the 0.33.0 value (88.6) and the repeats (90–106) sit in one
broad band, and the campaign cell sits outside it.

**Why the original evidence did not catch this.** Every argument above is
*within* one cell: 27 arms that shared a single machine state, one server
process, one thermal condition. Such evidence can establish that a cell ran
slow — it cannot separate "this cell ran slow" from "this build is slow",
because the confound is constant across every arm in it. The missing control
was the cheapest one available: measure the same thing twice. It takes 40
seconds and was not run until after the claim was written down.

A follow-up A/B compounded the error before it was caught. Serving the archived
0.33.0 binary against the current MLX payload (old Go, new MLX — the pairing
that is possible because a binary does not carry its payload, see BINARIES.md)
measured 89.0, which was read as "fast, therefore the MLX pin is exonerated and
the regression is Go-side". With 0.33.2 itself measuring 90–106, 89.0 is
unremarkable and that comparison establishes nothing. A wrong baseline
invalidates every reading taken against it.

**What this means for the numbers in §1 and §2.** The quality columns stand:
they reproduce across builds, and §1's agreement to three decimals is itself
evidence the measurement is sound. The `Gen tok/s`, `s/req` and `req/h` columns
are single-run figures on a host with ~2x state-dependent spread, so they are
usable as an order-of-magnitude characterisation of a model and **not** as a
build-to-build comparison. The same caveat applies retroactively to the
throughput columns in
[the 2026-08-28 campaign](vision-campaign-2026-08-28-mlx0330-nvfp4.md).

**What would make throughput comparable** is repetition, which the harness
already supports and this campaign did not use: `REPEATS=n` with
`summarize_reps.py`, which renders a spread rather than a point. Nothing
here should be read as evidence that 0.33.2 is faster or slower than 0.33.0.

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
- **No gate catches a throughput change, and §3 shows a gate would need
  repeats to mean anything.** Preflight asserts token accounting and payload
  identity, not speed. A single-run speed gate on this host would fire on
  machine state, not on code — which is exactly the mistake §3 records.
- **n=1 per cell, and §3 is what that cost.** Quality reproduces well enough
  to be convincing at n=1; throughput does not, and treating a single cell as a
  baseline produced a confident regression claim that repetition refuted. Use
  `REPEATS=n` for any campaign whose conclusions depend on timing.
- **Scores are on-host, untracked**
  (`docs/maxusai/vision-suite/scores_mlx0332nv1_*.json`, seven files), per the
  campaign convention; this document is the durable record.
