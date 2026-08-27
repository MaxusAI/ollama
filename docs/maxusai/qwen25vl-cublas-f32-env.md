# qwen25vl → GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 (launcher-gated, no payload patch)

**Status: mechanism CONFIRMED at runtime (2026-08-26, PR #214 comment
5421905441).** On stock `ollama/ollama:0.33.0` (latest release) the poison
class is alive (`HHXH` on the `02c9d7e1…` trigger) and
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` heals it (`HH`); the three-way A/B is
causal — `f32` and `bf16` heal, `f16` reproduces. `bf16` healing confirms
range-not-mantissa: cuBLAS bf16 GEMMs accumulate in fp32, so both settings
satisfy "≥fp32 accumulation".

**Dress rehearsal PASSED on both failing GPU generations** (PR #215 comments
5422486826 and 5422669657): a branch-built `/bin/ollama` over
`maxusai/ollama:sync-0.32.15` gives gate-default **HHH** on the poison probe
with `f32` verified in the runner env — Blackwell and Turing alike; operator
`=f16` wins and reproduces stock garbage (Blackwell sticky `HXX`, Turing
recovering `HXH`); operator `=bf16` wins and heals on both, including sm_75
(no native bf16 tensor cores — cuBLAS still takes the path), at timing
parity with f32 (Blackwell 676 vs 734 ms, Turing 1059 vs 1223 ms warm poison
encode, n=1 each). `f32` stays the gate default: closest to CPU numerics and
bf16 buys nothing measured; `bf16` remains a per-container operator choice
validated on every CUDA generation in the estate.

**DEPLOYED 2026-08-27** on `maxusai/ollama:sync-0.33.0`
(`0.33.0-dynres-0-g5171887`, main @ `51718870`): full CUDA preflight PASS
19/19 with `poison_probe` on the natively-gated binary, then vsuite
recreated with **no** workaround env vars — runner env shows the
gate-injected `f32` while the container env is clean, and the checkerboard
trigger decodes healthily in production. The interim global-f32 container
workaround (2026-08-26 ~11:49Z → 2026-08-27) is retired; scored cells from
that window carry global-f32 numerics.

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
  preserved (`=f16` reproduces stock behavior for A/B; `=bf16` also heals,
  measured on Blackwell and Turing both, at timing parity with f32).
- CUDA and ROCm both honor the knob (shared `ggml-cuda` source). CPU ignores
  it (already fp32-accumulate); Metal and Vulkan payloads do not read it —
  Apple serving is a separate numerics domain, covered by preflight.
- Perf: vision encode pays f32 GEMMs (weights dequantized to f32 +
  `cublasSgemm`). Text side: quantized tags are carried by MMQ/MMVQ and the
  f16 decode-vector kernels already accumulate fp32, so cost is ~nil;
  `-fp16` text tags pay an fp32-GEMM prefill cost. Measure through the
  standard harness arm, per the one-runner invariant.

## Validation

Probe shape and H/X decision table: `docs/maxusai/clip-mm-prec-f32-validation.md`.
Its step 0 (the zero-rebuild env test — this exact mechanism) is **done**:
measured healed on stock 0.33.0 with the three-way `f32`/`bf16`/`f16` A/B
(PR #214 comment 5421905441), resolving the decision table to "ship the
gate". The gated-fork-build dress rehearsal is also **done**, on both GPUs
(see the Status block above; PR #215 comments 5422486826 / 5422669657):
gate-default `HHH` with `f32` verified in the runner env, `=f16` override
wins and reproduces the garbage, `=bf16` override wins and heals. To verify
the gate on any future build: the subprocess debug log line
(`llm/llama_server.go` logs `GGML_*` keys at startup) or
`/proc/<runner-pid>/environ`.

## Pin-bump hazard

The knob is an env var in ggml's CUDA backend, not API. On every
`LLAMA_CPP_VERSION` bump, confirm the new pin still reads
`GGML_CUDA_CUBLAS_COMPUTE_TYPE` in `ggml/src/ggml-cuda/ggml-cuda.cu`
(`llama/README.md` review checklist carries this). If upstream removes or
renames it, fall back to shipping patch 904 from `fix/clip-mm-prec-f32`.
