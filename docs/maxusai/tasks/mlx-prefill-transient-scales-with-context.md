# TASK: the MLX prefill transient scales with context — price it per rung, bound the image overlay

**Opened:** 2026-09-06. **Status:** OPEN, unassigned. **Found by:** the session-growth probe
of the `main` validation (Glenn's item 3, "what a deploy of a `main` build should know").
**Evidence:** the probe records are in-repo under `vision-suite/preflight/runs/` —
`gpu276-session-growth-2026-09-06.jsonl`, `gpu276-growth-shape-2026-09-06.jsonl`,
`gpu276-slopes-2026-09-06.jsonl` (31b and qwen3.8 text slopes, measured the same evening); the
runner logs (`session-growth-runner.log`, `growth-shape-runner.log`, `trace-arrays-runner.log`)
stay on the CUDA host in `preflight-runs/`. All on the CUDA host, build
`main-a523d60b` (= `sync-0.33.3` payload, #276's admission, 16 GiB `OLLAMA_GPU_OVERHEAD`
except where noted).

## What was found

#276 prices a load as weights + KV(`num_ctx`) + a per-architecture headroom calibrated on
2026-09-05 from one- and three-image requests with short prompts. That price holds for such a
request. It does not hold for a *conversation* that fills the rung:

- **gemma4:31b at an explicit 65536** (priced 37.6 GiB against a 39.9 GiB budget): a chat with
  three new images per turn peaked 28.4 → 41.5 → **49.1 GiB and died** (`cudaMallocAsync …
  out of memory`) at turn 3 — 12.5k tokens, 20 % of the rung. **qwen3.8** (priced 31.6) climbed
  to **44.8 GiB** at 55k tokens and survived only because the card had room; the MLX limit is
  soft.
- Resident arrays do not grow: at request end the runner holds weights + KV + drafts (8.9 GiB
  on 12b at 8.8k tokens, trace dump). The growth is a **transient inside each request**.

## The law (gemma4:12b, weights 7.2 GiB; `probe-growth-shape.sh`)

| shape | context | peak | reading |
|---|---|---|---|
| fresh **text-only** prompt | 5.2k / 10.4k / 20.9k / 41.6k | 16.7 / 19.9 / 23.3 / 28.8 GiB | transient ≈ 8 GiB + **~0.3 GiB per 1k tokens**, linear to the rung (the priced KV is 0.02/1k) |
| 3 images in turn 1, then text turns | 3.3k → 23.8k | 22.0 → 25.1 → 27.0 → 27.9 → 29.4 GiB | a fixed image offset, then the text slope — images resting early in the history cost nothing extra |
| 3 images at the **start** of one 24k-token prompt | 24.2k | 30.7 GiB, served | the overlay widens only to the end of the **last** image block (ADR 0014) |
| **new images appended every turn** (what chat clients do) | 4.4k / 8.8k / 13.2k | 16.4 / 23.9 / 30.6 GiB; turn 4 refused | the last block sits at the end, the opening chunk widens to the whole prompt, the dense overlay is ∝ prompt²: **+1.7 GiB per 1k** on 12b, **+2.5** on 31b |

Mechanism, from the code and the runner's own refusal (`image request needs a 1.0 GiB
attention mask (16596-token prompt); reduce the prompt length or the image-token budget`,
`x/mlxrunner/media.go:205-245`, ADR 0014 `81a517a3`): bidirectional media (gemma4 images)
prefills with a dense float32 mask over an opening chunk that *"grows to cover every image
block"* (`extendChunk`); the mask is chunk × promptLen × 4 B and the attention scores over
that widened chunk are quadratic in the prompt. The guard caps only the **mask** at 1 GiB
(~16k tokens when the chunk is the prompt) — the **scores** are unpriced and larger, which is
why 31b died at 12.5k, below the guard. Causal media (the qwen3.5 family) is exempt and shows
only the linear term. The linear text term is un-attributed but reproducible (a per-chunk
attention buffer over the context is the obvious candidate; the prefill chunk is a hard-coded
2048, `pipeline.go:22`).

### Text-only slopes, measured 2026-09-06 (`probe-slopes.sh`; `gpu276-slopes-2026-09-06.jsonl`)

Same shape as leg A: one fresh text prompt per request, explicit `num_ctx` 65536, `num_predict`
32, `OLLAMA_GPU_OVERHEAD=0`, one model at a time, beside production; the peak is the runner's
`peak memory` line. Weights = the 16-token warm-up peak.

| model | weights | peak at 5.2k / 10.4k / 20.9k / 41.6k tokens | fit over the ≥ 10k points | at a full 65536 | #276 prices |
|---|---|---|---|---|---|
| gemma4:12b-nvfp4 (leg A above) | 7.2 GiB | 16.7 / 19.9 / 23.3 / 28.8 GiB | 17.2 + **0.28** GiB per 1k tokens | ≈ 36 GiB | 22.9 GiB |
| gemma4:31b-nvfp4 | 17.6 GiB | 30.2 / 37.3 / 43.4 / 55.3 GiB | 31.3 + **0.58** GiB per 1k | ≈ 69 GiB | 37.6 GiB |
| qwen3.8:27b-nvfp4 | 17.4 GiB | 28.3 / 30.3 / 35.1 / 41.0 GiB | 27.3 + **0.33** GiB per 1k | ≈ 49 GiB | 31.6 GiB |

- The gemma4 opening segment (5k → 10k) is steeper — 0.62 on 12b, **1.36** on 31b — and
  settles to the slope above; qwen3.8 is linear from the start (0.39, then 0.33). The ≥ 10k
  fit is the large-context slope; a fit over all four points gives 0.32 / 0.65 / 0.35 and lands
  about 2 GiB higher at 65536.
- The intercept minus the weights — the transient at zero context — is 10.0 / 13.7 / 9.9 GiB:
  the constants ADR 0034 calibrated (gemma4 14.5, qwen3.5 dense 10.5) are the `a` of the line,
  as its amendment says.
- Within an architecture the slope tracks the weights: 0.28 / 7.2 = 0.039 and 0.58 / 17.6 =
  0.033 GiB per 1k tokens per GiB of gemma4 weights (n = 2); 0.019 for qwen3.5 dense (n = 1).
  That is why fix 1 below can be one constant per architecture times the weights rather than a
  per-model table.
- Wall time is not a memory result, but for the record: the first sized prompt on each model
  was slow once (781 s for 5.2k tokens on 31b, then 99 s for 10.4k); qwen3.8 prefilled 41.6k
  tokens in 31 s.

## Why the calibration missed it

Every 2026-09-05 calibration request had ≤ 3.4k prompt tokens, so the peaks were flat across
rungs and the headroom became a constant. ADR 0034 states that flatness; it is true only for
short prompts and needs the qualifier (amended alongside this task).

## Production exposure

A gemma4 chat that keeps adding images and reaches ~8–12k tokens of context on 26b/31b dies
mid-prefill under a ~40 GiB budget; past ~16k the mask guard refuses it cleanly. Text-only
sessions that fill a 65536 rung need ≈ 36 GiB on 12b (priced 22.9), ≈ 69 GiB on 31b (priced
37.6) and ≈ 49 GiB on qwen3.8 (priced 31.6); the 31b figure is more than any budget this
shared card offers, so an explicit 65536 on 31b is admitted today and cannot be filled.
Single-image requests (the teacher-v3 loop) never approach either.

## Fixes, in layers

1. **Headroom that scales with the rung.** `admissionHeadroom` becomes `a_arch + b_arch ×
   num_ctx` with `b` from the text-only slopes above (≥ 10k fit: 12b 0.28, 31b 0.58, qwen3.8
   0.33 GiB per 1k tokens), in the one-constant form `b_arch × weights × num_ctx` (gemma4
   0.035, qwen3.5 dense 0.019 GiB per 1k per GiB of weights; the MoE is unmeasured — take
   gemma4's until it is). Honest and small; it will refuse or clamp big rungs on this card where the
   constant admitted them, which is the point (GGML's fit-derived default does the same).
2. **Per-request pricing of the dense overlay.** The ADR 0014 guard should charge the widened
   chunk's *scores* (chunk × promptLen × heads × bytes for the widest layer), not the mask, against
   the remaining budget, and keep its actionable message. `x/mlxrunner/media.go`, unit-testable.
3. **Bound the dense region** — ADR 0014's own deferred follow-up: make `visionChunkMask`
   offset-aware and the bidirectional branch cache-history-aware so the dense span is one image
   block, not the prompt. The real fix for image chats; a gemma4 model-layer change.
4. Optional: a smaller prefill chunk for long contexts trades throughput for the linear term.

## Acceptance criteria

1. ☑ Text-only slope measured for gemma4:12b, 31b and qwen3.8 (2026-09-06, table above).
   ☐ The headroom formula reproduces the leg-A peaks within ~10 % at 4k–42k.
2. ☐ A conversation that fills a 65536 rung with text on 12b runs to the rung without OOM under
   the priced need (the priced need now covers it).
3. ☐ The image-append chat shape is refused per request with the guard's message *before* the
   scores are allocated, on 31b under a 40 GiB budget — no mid-prefill abort.
4. ☐ After fix 3: the same chat shape on 31b runs to ≥ 32k tokens with a peak within the priced
   need; the mask guard's ceiling can then be lifted as ADR 0014 intended.
5. ☐ No new over-refusal on the vision-suite cells (all prompts ≤ 3.4k tokens): T1 on the five
   nvfp4 models still 27/27 at 8192.

## Not in scope

The GGML path (its KV and compute buffers are preallocated from `num_ctx`/`num_batch`; no
equivalent transient), and the qwen3.5 linear term's root cause beyond measuring its slope.
