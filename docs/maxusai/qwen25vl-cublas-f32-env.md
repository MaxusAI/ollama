# qwen25vl → GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (launcher-gated, no payload patch)

**What:** when the launcher starts a llama-server runner for arch `qwen25vl`,
`applyArchServerEnvs` (`llm/llama_server.go`) sets
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` in that subprocess's environment — unless
the operator already set the variable, in which case their value wins.

**Why:** PR #214 localized the qwen2.5vl:3b poison-image garbage (`'?'×31`,
poisoned slot until reload, quant-independent) to the one fp16 stage every
failing CUDA config shares and the healthy CPU path lacks: the vision
tower/merger's f16-weight matmuls run as fp16-accumulate cuBLAS GEMMs
(`compute_type = src0->type` in `ggml_cuda_mul_mat_cublas` at pin b10488),
while CPU always accumulates fp32. `=f32` is stock ggml's process-wide
compute-type override — and since each runner serves exactly one model,
process-wide is model-scoped here. Diagnosis:
`docs/maxusai/qwen25vl-3b-poison-image-garbage-decode.md`.

**Why this shape instead of a payload patch:** the sibling branch
`fix/clip-mm-prec-f32` prepares compat patch 904 (per-op `GGML_PREC_F32`
post-pass over the clip graph) with runbook
`docs/maxusai/clip-mm-prec-f32-validation.md`. The launcher gate is preferred
for shipping because:

- **No fork-carried llama.cpp patch** — nothing to re-validate or re-cut on
  every `LLAMA_CPP_VERSION` bump.
- **Default behavior unchanged for every other model** — no vision-baseline
  drift for gemma4/nemotron/qwen3vl, no preflight re-recording outside the
  qwen25vl family.
- **Go-only change** — deployable via the Go binary swap; the knob already
  exists in every shipped CUDA/ROCm payload at the current pin.
- **Probe = fix**: the zero-rebuild hypothesis test in the 904 runbook (step
  0) sets exactly this variable, so a passing probe is a dress rehearsal of
  the shipped mechanism.

Patch 904 stays shelved as the upstreamable form and the fallback if the env
knob disappears from a future pin.

## Semantics and scope

- Applies to arch `qwen25vl` only (all sizes: 3b/7b/32b). qwen2vl shares the
  clip graph builder but has no measured trigger — extend the switch if one
  shows up. The class is latent per-graph (see the diagnosis doc's
  correction: each implementation has its own poison set), so family-wide is
  deliberate; if 7B baseline continuity outweighs latent risk, the gate can
  key on `qwen25vl` + embedding width 2048 to hit only the 3B.
- Operator override: any inherited `GGML_CUDA_CUBLAS_COMPUTE_TYPE` value is
  preserved (`=f16` reproduces stock behavior for A/B; `=bf16` is the
  range-safe middle ground worth one probe cell).
- CUDA and ROCm both honor the knob (shared `ggml-cuda` source). CPU ignores
  it (already fp32-accumulate); Metal and Vulkan payloads do not read it —
  Apple serving is a separate numerics domain, covered by preflight.
- Perf: vision encode pays f32 GEMMs (weights dequantized to f32 +
  `cublasSgemm`). Text side: quantized tags are carried by MMQ/MMVQ and the
  f16 decode-vector kernels already accumulate fp32, so cost is ~nil;
  `-fp16` text tags pay an fp32-GEMM prefill cost. Measure through the
  standard harness arm, per the one-runner invariant.

## Validation

Probe shape and H/X decision table: `docs/maxusai/clip-mm-prec-f32-validation.md`
(step 0 is this exact mechanism; step 2's causality check becomes
`-e GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` on a gated build → garbage must
return). Verify the gate is live in the runner env via the subprocess debug
log line (`llm/llama_server.go` logs the filtered env at startup) or
`/proc/<runner-pid>/environ`.

## Pin-bump hazard

The knob is an env var in ggml's CUDA backend, not API. On every
`LLAMA_CPP_VERSION` bump, confirm the new pin still reads
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` in `ggml/src/ggml-cuda/ggml-cuda.cu`
(`llama/README.md` review checklist carries this). If upstream removes or
renames it, fall back to shipping patch 904 from `fix/clip-mm-prec-f32`.
