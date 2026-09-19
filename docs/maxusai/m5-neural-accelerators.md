# The M5 Neural Accelerators, and which of the fork's paths reach them

**Measured 2026-09-19 on `10.8.0.3` — Apple M5 Max, 18 cores, 128 GB, macOS 26.6.2,
Metal compiler 32023.921 — against the deployed build `0.34.0-maxusai-8a7ba949`
(llama.cpp `b10864`, MLX `d9add9d1` = `0.32.2-61-gd9add9d`).**

The M5 GPU adds per-core Neural Accelerators: dedicated matrix-multiply hardware,
reached from Metal 4 through `mpp::tensor_ops` (Metal Performance Primitives).
Apple's own write-up
([Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU](https://machinelearning.apple.com/research/exploring-llms-mlx-m5))
states MLX needs **macOS 26.2 or later** to use them. This host is 26.6.2.

**The question this answers is not "should we adopt it" — we already ship it.**
It is which of the fork's paths reach the accelerators, and what they are worth.

## 1. What the shipped payload already contains

Read out of the deployed artifact, not inferred:

| artifact | evidence |
|---|---|
| `mlx.metallib` | compiled for `air64_v28-apple-macosx26.2.0`; 2564 `tensor_ops` / `simdgroup_matrix` hits; `mpp::tensor_ops::matmul2d`, MLX's `NAXTile` / `BaseNAXFrag` |
| `libmlx.dylib` | `MTL4`, `MTLTensor` symbols |
| `llama-server` | `GGML_METAL_HAS_TENSOR` compiled in, plus the `GGML_METAL_TENSOR_ENABLE` / `_DISABLE` runtime knobs |

`supportsFamily:MTLGPUFamilyMetal4` (enum 5002, which ggml hardcodes because older
SDKs lack the symbol) answers **YES** on this device — probed directly with a
20-line Metal program, not read off a log.

### The MLX kernel inventory is uneven, and nvfp4 is the thin side

Every `nvfp4` pipeline in the metallib, by family. Only two carry `nax`:

```
nvfp4_gather_qmm_rhs_nax_nn / _nt         <- Neural Accelerator (the gather / MoE path)
nvfp4_qmm_n, nvfp4_qmv{,_fast,_quad,_wide},
nvfp4_qvm{,_split_k}, nvfp4_gather_qmm_n,
nvfp4_gather_qmv, nvfp4_gather_qvm        <- no nax
```

The dense `nax` matmuls that do exist — `affine_qmm_t_nax`, `affine_qmm_n_nax`,
`qmm_t_nax_tgp_impl` at `gs_128` — are all **affine** quantisation. Upstream
disabled the non-transposed NAX qmm and fixed `group_size < 64` separately
(ml-explore/mlx#4202, already in our pin); **nvfp4 is group 16**, squarely in that
regime.

Our pin is not stale: it is dated 2026-09-14 and is itself a NAX commit (#4481).
Only five MLX commits have landed since, none of them Metal-NAX work.

## 2. GGUF: the accelerators are already on, and worth 2.14x prefill

`b10864` gates the tensor API in four steps (`ggml/src/ggml-metal/ggml-metal-device.m`):
`supportsFamily:Metal4`, then `GGML_METAL_TENSOR_DISABLE`, then a **device-name
allowlist** (`M5` / `M6` / `A19` / `A20`, bypassable with `GGML_METAL_TENSOR_ENABLE`),
then a runtime compile of a dummy `mpp::tensor_ops::matmul2d` kernel. Upstream's own
comment explains the allowlist: the tensor path was *"~5% slower"* on M2 Ultra and
*"no significant difference"* on M4/M4 Max. "Apple M5 Max" passes it.

`gemma4:31b-it-q4_K_M`, a ~4000-token prompt, `num_predict=1`, `num_ctx=8192`, one
cold server per arm, model warmed before timing, **a different prompt of the same
length on every rep**:

| arm | median prefill | reps |
|---|---|---|
| default | **275.2 tok/s** | 277.5 / 279.8 / 272.9 / 268.9 |
| `GGML_METAL_TENSOR_DISABLE=1` | **128.6 tok/s** | 124.3 / 130.5 / 127.2 / 129.9 |
| `GGML_METAL_TENSOR_ENABLE=1` | 261.3 tok/s | 257.6 / 264.7 / 261.0 / 261.6 |

**2.14x.** `ENABLE` matching default is the expected no-op: it only skips the name
check this device already passes, and the 5% gap against default is between-restart
variance, against ~2% within an arm.

**Two traps, both of which produced wrong numbers first.** Re-sending an identical
prompt is served from the **prefix cache** at 16000-34000 tok/s, which is not prefill
at all; and the first call after a load JIT-compiles pipelines, so an unwarmed first
rep reads ~2x slow. The published llama.cpp-on-M5 report of a failing tensor-API
check does **not** reproduce here.

**Unexplained, and recorded as such:** the binary contains all three tensor-API log
strings (`testing tensor API for f16 support`, `tensor API disabled for pre-M5`,
`the tensor API is not supported in this environment`) and the gate should emit one
of them, but no arm logs any. The measurement does not depend on the logs.

## 3. MLX: affine int4 is ~11% faster than nvfp4 at the prefill chunk

`qqmm` grew `affine64` / `affine128` methods for this comparison, so the affine arms
pay the same setup as the nvfp4 one and the timing compares kernels rather than
plumbing. Generated output, pasted verbatim (ADR 0012 rule 8):

# qqmm bench: median wall ms per call (TFLOP/s) after warm-up; relRMS vs fp32 · unquantized w

## gemma4-31b q_proj (K=5376, N=8192)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | affine64 ms (TFLOP/s) | affine128 ms (TFLOP/s) | qmm / bf16 | qmm / affine64 | qmm / affine128 | relRMS bf16 / qmm / affine64 / affine128 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.377 (0) | 0.308 (0) | 0.279 (0) | 0.247 (0) | 0.82x | 1.11x | 1.25x | 2.8e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 512 | 2.389 (19) | 3.077 (15) | 2.925 (15) | 3.174 (14) | 1.29x | 1.05x | 0.97x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 2048 | 9.338 (19) | 10.910 (17) | 10.725 (17) | 10.417 (17) | 1.17x | 1.02x | 1.05x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |

## gemma4-31b o_proj (K=8192, N=5376)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | affine64 ms (TFLOP/s) | affine128 ms (TFLOP/s) | qmm / bf16 | qmm / affine64 | qmm / affine128 | relRMS bf16 / qmm / affine64 / affine128 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.311 (0) | 0.179 (0) | 0.205 (0) | 0.204 (0) | 0.57x | 0.87x | 0.87x | 2.9e-03 / 1.0e-01 / 9.0e-02 / 1.0e-01 |
| 512 | 2.025 (22) | 2.215 (20) | 2.359 (19) | 2.439 (18) | 1.09x | 0.94x | 0.91x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 2048 | 8.005 (23) | 9.162 (20) | 9.276 (19) | 9.351 (19) | 1.14x | 0.99x | 0.98x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |

## gemma4-31b gate_proj (K=5376, N=21504)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | affine64 ms (TFLOP/s) | affine128 ms (TFLOP/s) | qmm / bf16 | qmm / affine64 | qmm / affine128 | relRMS bf16 / qmm / affine64 / affine128 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.809 (0) | 0.433 (1) | 0.318 (1) | 0.358 (1) | 0.54x | 1.36x | 1.21x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 512 | 6.522 (18) | 8.150 (15) | 8.017 (15) | 7.750 (15) | 1.25x | 1.02x | 1.05x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 2048 | 22.765 (21) | 24.734 (19) | 22.652 (21) | 21.112 (22) | 1.09x | 1.09x | 1.17x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |

## gemma4-31b down_proj (K=21504, N=5376)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | affine64 ms (TFLOP/s) | affine128 ms (TFLOP/s) | qmm / bf16 | qmm / affine64 | qmm / affine128 | relRMS bf16 / qmm / affine64 / affine128 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.801 (0) | 0.394 (1) | 0.376 (1) | 0.383 (1) | 0.49x | 1.05x | 1.03x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 512 | 8.113 (15) | 8.938 (13) | 8.211 (14) | 8.494 (14) | 1.10x | 1.09x | 1.05x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |
| 2048 | 26.735 (18) | 30.030 (16) | 28.624 (17) | 26.162 (18) | 1.12x | 1.05x | 1.15x | 2.9e-03 / 1.0e-01 / 9.1e-02 / 1.0e-01 |

### Reading those tables

Summed over the four projections at the **2048-row prefill chunk**, two independent
runs an hour apart:

| | run A | run B |
|---|---|---|
| nvfp4 (`qmm`) | 81.40 ms | 74.84 ms |
| `affine128` | 72.40 ms | 67.04 ms |
| **nvfp4 / affine128** | **1.124x** | **1.116x** |

Absolute times moved ~10% between runs; **the ratio did not**, and it holds per
shape (`qmm/affine128` at M=2048: q_proj 1.09/1.05, o_proj 0.97/0.98, gate_proj
1.16/1.17, down_proj 1.18/1.15). So affine int4 is **~11-12% faster than nvfp4
across the prefill chunk**, consistently, with `o_proj` the one shape where nvfp4
is marginally ahead. Absolute numbers from a single run of this bench should not be
quoted; ratios within a run should.

At **decode** (M=1) there is no consistent winner — that regime is memory-bound, and
both are 4-bit.

**bf16 is the fastest of all at prefill** (66.84 ms in run B, 1.12-1.17x ahead of
nvfp4), which is the dense NAX GEMM doing exactly what it should — at 4x the memory.

**This is a smaller gap than the kernel inventory predicts.** §1 shows dense NAX
matmuls for affine and none for nvfp4, which should be worth far more than 11%.
Either nvfp4's dense path is less disadvantaged than the pipeline names suggest, or
affine is not dispatching to its NAX kernel at these shapes. **Dispatch was not
confirmed**, and nothing here should be read as having confirmed it.

## Limits

- Synthetic random-normal weights. The error columns (nvfp4 `1.0e-01`, affine64
  `9.1e-02`) describe *these* operands, not a real checkpoint, and must not be read
  as a quality ranking — the 2026-09-18 OCRBench ladder has accuracy *decreasing*
  monotonically as precision rises, on two backends.
- Medians over 10 timed calls after 3 warm-ups, one run per cell, on an otherwise
  idle GPU. Anything measured beside other GPU work is measuring contention.
- The GGUF arm is one model, one prompt length, one quantisation.

## Follow-ups

1. **Guard the GGUF tensor path — `vision-suite/preflight/nax_probe.m`, written for
   this.** It replicates the gate's own decision (family, `DISABLE`, name allowlist,
   dummy-kernel compile) in ~100 ms cold and ~2 ms warm, without loading a model, and
   exits 0/1. On this host:

   ```
   {"device":"Apple M5 Max","supports_metal4_family":true,"name_allowlisted":true,
    "dummy_kernel_compiles":true,"pipeline_ms":96.1,"has_tensor":true}
   ```

   which agrees with the 2.14x measurement by a route that shares nothing with it.
   `GGML_METAL_TENSOR_DISABLE=1` flips it to `has_tensor:false`, exit 1.

   **The kernel it compiles is copied verbatim from ggml and must stay in sync.**
   A hand-written reconstruction failed to compile here — wrong descriptor arity,
   wrong template arguments, a missing `tensor_inline` tag — and would have raised a
   false alarm against a perfectly healthy host. The file carries the one-liner that
   re-extracts it from the pinned llama.cpp.

   Still to do: wire it into `preflight.py` as a check, with an expectations field so
   a host that *should* accelerate and doesn't fails the gate rather than merely
   printing. Pair it with a `strings` assertion that the built `llama-server` carries
   `GGML_METAL_HAS_TENSOR`, so the build half is covered too.
2. **Ask upstream about NAX for `group_size < 64`.** That single change would move the
   whole nvfp4 fleet; ml-explore/mlx#4202 is the thread.
3. **An ~11% prefill gain is probably not worth a requantisation on its own**, and it
   is not separable from the quality question, which does not order by precision.
