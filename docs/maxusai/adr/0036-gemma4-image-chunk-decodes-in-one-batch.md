# ADR 0036: a gemma4 vision runner decodes its image chunk in one batch

- **Status:** accepted 2026-09-18 (Glenn: "raising it to the 1120 ceiling"). Sits on
  [ADR 0008](0008-gemma4-budget-fill-restores-1120.md), which put the 1120 ceiling
  there in the first place.
- **Date:** 2026-09-18
- **Deciders:** MaxusAI fork maintainers

## Context

The scheduler derives llama-server's generation batch from the context rung (`generationBatchForContext`: 512
up to 4096, 1024 above, 2048 above 32768), and the fork's launcher passes it as both `-b` and `-ub`
(`appendBatchArgs`). gemma4's image tokens attend to each other, and `mtmd_helper_decode_image_chunk` splits an
image chunk larger than `n_batch` into `n_batch`-sized decodes that are bidirectional only within themselves. At
the fork's default ceiling of 1120 image tokens a top-rung gemma4 image is 1064–1121 tokens, so at every rung the
vision suite serves (8192, batch 1024) it has been decoded as `1024 + 40–97`: the first 1024 image tokens never
attend to the tail. Upstream hit the same geometry from the other side: llama.cpp #28954 aborts llama-server
outright when a non-causal chunk exceeds `n_ubatch`, because its launcher keeps `-ub 512`.

Measured on the deployed 0.34.1 image (`preflight-runs/batchab-0341-render-thinkfalse.md`, task doc
`upstream-sync-0.34.1.md` § "The gemma4 image resize algorithm"): the two gemma4 GGUF models that decode images
non-causally, think-off at the 8192 rung, two repeats per arm, every scored cell identical between repeats.

| | 31b, batch 1024 | 31b, batch 2048 | 26b-a4b, batch 1024 | 26b-a4b, batch 2048 |
|---|---|---|---|---|
| scene bbox IoU | 0.966 | 0.963 | 0.978 | 0.977 |
| contracts, invoice, name_bbox | identical | identical | identical | identical |
| fine text 22/16/12/9/7 px | 4/4/4/**4**/3 | 4/4/4/**3**/3 | 4/4/4/3/3 | 4/4/4/3/3 |
| prefill tok/s | 474 / 508 | 753 / 721 | 1250 / 1369 | 2434 / 2518 |
| s/req | 12.7 / 12.6 | 11.5 / 11.6 | 4.5 / 4.2 | 3.7 / 3.7 |

The split costs nothing the scored cells see. The single piece is 1.5× (31b) and 1.9× (26b-a4b) faster at
prefill and moves one knife-edge cell, 31b's 9 px fine-text tier, 4 → 3, deterministically — the same shape as
the Metal encoder finding in #312, where the corrected kernel also scores one 9 px tier lower on 31b. On the ROCm
host (#313) the same rung decodes past `n_ubatch` too, and the pixel geometry is identical digit for digit.

## Decision

A gemma4 runner with the vision capability (the scheduler's own `CheckCapabilities` rule: a projector layer, or the tower
inside the main GGUF as the fork's gemma4 checkpoints carry it) starts its automatic generation batch from the smallest rung at or above its
resolved image-token ceiling — 2048 at the default 1120, 1024 at a request pinned to 560, 512 at 280 — instead
of the context rung, and only steps down when that batch does not fit the memory left after the model
(`server/sched.go`: `imageChunkGenerationBatch`, `automaticGenerationBatch`'s `floor`). An explicit `num_batch`
still wins; embedding loads and the constrained-CUDA-without-flash-attention path are untouched.

## Options considered

- **Leave the split.** No measured contract cost, so defensible; but it leaves the 9 px tier on the
  less-faithful path and 1.5–1.9× of prefill on the table for the fork's own default ceiling.
- **Raise the default to 2048 for every vision runner.** Pays the 2 GiB compute-buffer surcharge on arches that
  gain nothing from it (qwen-VL and the other causal image decoders), and on every other vision load.
- **Fit the chunk to the batch upstream's way** (llama.cpp #28954's fix, when it lands): retires this ADR.

## Consequences

- The generation batch for gemma4 vision at the 8192 rung is 2048, with `generationBatchSurcharge`'s 2 GiB
  budgeted at admission; on a card without that headroom the batch steps down and the split returns, logged as
  "generation batch below the image chunk".
- gemma4 e2b and e4b decode images causally since llama.cpp #28335, so the split never hurt them; they get the same
  batch because the family and the vision capability are what the scheduler can see, and their compute buffers are small.
- `nemotron_h_omni`'s ceiling (3328) exceeds the batch ladder; it stays on the context rung, out of scope here.
- The MLX path is unaffected; it never goes through llama-server.
- Retirement: llama.cpp fits a non-causal image chunk to one ubatch, or sizes ubatch to the image cap
  (retirement register, "gemma4 image chunk vs. batch").
