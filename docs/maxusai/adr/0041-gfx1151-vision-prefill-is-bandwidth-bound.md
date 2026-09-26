# ADR 0041: gfx1151 vision prefill is bandwidth-bound, so ROCm/HIP configuration is closed

- **Status:** accepted 2026-09-21. Closes the question opened at the end of the
  MMQ vs hipBLAS A/B and pursued in
  [rocblas-on-gfx1151.md](../rocblas-on-gfx1151.md). Sits beside
  [ADR 0040](0040-rocm-10-is-experimental-until-it-is-faster.md), which declined
  ROCm 10.0.0 on measurement.
- **Date:** 2026-09-21
- **Deciders:** MaxusAI fork maintainers

## Context

After measuring that forcing hipBLAS cost MoE prefill up to 35.6%, I wrote that
rocBLAS was worth investigating for "alternative more optimized kernels". This
ADR records what that investigation found, so it is not re-opened on the same
reasoning.

Eight levers were tested on gfx1151, build `0.34.2-dynres-f67b1aef`, ROCm 7.2.4,
against a +-0.4% throughput noise floor:

| lever | result |
|---|---|
| hipBLASLt (`ROCBLAS_USE_HIPBLASLT=1`) | inert, -0.8%. Neither the ROCm 7.2.4 nor the 10.0.0 image ships a single gfx1151 hipBLASLt kernel file |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | -48.6% throughput, **no** accuracy gain |
| `prefer_f32_output` for RDNA3.5 | -30.2% / -33.3%, **no** accuracy gain |
| Q4_K MMQ `default: return true` | -0.1% on dense `qwen3.8:27b` |
| rocWMMA flash attention | the option no longer exists at b10969 |
| MMQ tile tuning (upstream #21284) | superseded by `mmq-config-rdna3-5.cuh` |
| `GGML_HIP_NO_VMM`, `GGML_HIP_MMQ_MFMA` | already default **ON** |
| flash-attention kernel (MMA vs TILE) | not resolvable; see below |
| rocBLAS efficiency vs hardware ceiling | **82-87% of peak** on tower shapes |

The flash-attention row is worth stating precisely, because the selector looks
alarming at first read. The AMD WMMA branch in `ggml_cuda_get_best_fattn_kernel`
excludes head dims 40 and 72 outright and caps at 256, so **both gemma4 models
run entirely on the generic TILE kernel** -- the tower because 1152/16 = 72 is
on the exclusion list, the LM because 512 exceeds the cap. Forcing TILE on
models that would otherwise get MMA (compat 805, env-gated, gate verified to
fire: 0 log lines unset, 2 when set) measured:

| arm | MMA | TILE forced | delta |
|---|---|---|---|
| `nemotron3:33b` | 377.2 | 372.6 | -1.2% |
| `qwen3.6:35b-a3b` | 464.6 | 458.2 | -1.4% |
| `gemma4:31b` (**control**, already TILE) | 154.2 | 151.5 | **-1.8%** |

The control moved more than the treatment. gemma4:31b cannot be affected by the
gate, so its -1.8% is run-to-run drift between arms, and both treatment effects
are smaller than that. MMA vs TILE is not resolvable at this magnitude. The
head-dim exclusions cost nothing measurable here, and the 51% that flash
attention is worth on `gemma4:31b` is the TILE kernel's value, not the MMA
kernel's.

The rocBLAS efficiency row is the one that decides it. Every other row says "this
configuration is not better"; that row says there is little left to be better
than. A GEMM library at 82-87% of theoretical peak is not the bottleneck.

Where the time actually goes, per image request on `gemma4:31b`:

| block | cost |
|---|---|
| tower attention | ~12.2 TFLOP, **fp32**, memory-bound at 35.5 FLOP/byte against a ridge point of 116 |
| tower projections | 10.8 TFLOP, rocBLAS at 83-87% of peak |
| LM GEMMs | 8.7 TFLOP, rocBLAS at 43% (narrow, n~1100) |

Flash attention is worth **51%** of prefill here, measured with a three-arm test
that separated it from the KV-cache change it requires. It is doing real work,
not masking an underperforming path.

## Decision

**Stop pursuing ROCm/HIP kernel configuration for gfx1151 vision prefill.** The
production configuration is correct on every axis tested, and the remaining cost
is memory bandwidth and model shape, neither of which a build flag reaches.

This does not close:

- **Model-shape work** -- fewer image tokens, a smaller tower, or a quantised
  tower all reduce the bandwidth bill directly. That is where headroom lives.
- **A future ROCm shipping gfx1151 hipBLASLt kernels.** Re-test then; the
  mechanism is packaging, not architecture.
- **Upstream's MMA flash-attention kernel**, which replaced the rocWMMA path.
  If its gfx1151 tuning changes, re-measure with `gemm_ceiling_bench`, the
  three-arm FA test and compat 805.
- **Speculative decoding.** This ADR was written about prefill and that is a
  gap in it. Dense *decode* on this host runs at 75-85% of memory bandwidth
  (`gemma4:31b` 19.9 GB x 9.7 tok/s = 193 GB/s of 256; `qwen3.8:27b` 218 GB/s),
  so it is bandwidth-bound in the same way -- but speculative decoding is the
  one technique that beats that bound, by verifying several drafted tokens per
  weight read instead of one. It does not reduce bandwidth, it amortises it.
  llama-server at b10969 supports `--spec-type draft-dflash` and ollama passes
  it through (`llm/llama_server.go`), with drafts folded into the target's
  manifest (`create/draft.go`). Unmeasured here, and every published DFlash
  benchmark found so far is NVIDIA or vLLM.

## Consequences

- A proposal to change ROCm/HIP build flags or kernel dispatch for this hardware
  should cite a measurement against this table, not a general recommendation.
  Community guidance for "RDNA3" repeatedly proved wrong for RDNA3.5
  specifically -- upstream #24437 measured rocWMMA at -41% on gfx1151 while the
  same flag was being recommended broadly.
- `gemm_ceiling_bench.cpp` is committed as the instrument that answers "is this
  near the metal", which no comparative A/B can answer.
- The -48.6% compute-type figure is a **regression sentinel**: if prefill on this
  host ever halves, check `compute_type` before anything else.
