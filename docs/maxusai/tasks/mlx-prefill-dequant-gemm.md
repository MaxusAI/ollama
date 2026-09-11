# TASK: MLX nvfp4 prefill — the dequantise-to-bf16 GEMM opt-in, and the FP4 GEMM question

**Opened:** 2026-09-08. **Status:** code landed as an opt-in (off by default), **not yet
GPU-verified in the runner**. **Origin:** Glenn asked whether "NVGEMM" is used. Nothing by that
name exists in the fork, llama.cpp b10760 or MLX 37c26e57; what does exist is the question
behind it — does NVIDIA's native FP4 GEMM buy anything on the RTX PRO 6000 — and a cheaper
answer that fell out of measuring it.

## What runs the nvfp4 matmuls today

`QuantizedLinear.Forward` calls `mlx_quantized_matmul` (bf16 activations, nvfp4 weights). On
compute capability 12.0 MLX dispatches that to `qmm_sm80` for 8 rows and up — a CUTLASS/cute
mixed-input kernel that dequantises the fp4 weights in registers and runs bf16 tensor-core MMA,
JIT-compiled through NVRTC — and to `fp_qmv` for fewer rows. Dense bf16 matmuls go to cuBLASLt.
MLX also carries `qqmm` (`mlx_qqmm`): activations quantised to nvfp4 on the fly and the
cuBLASLt block-scaled FP4 GEMM, gated on compute capability 10 and up. Nothing in the fork
calls it. The llama.cpp path (the GGUF fleet) uses MMQ integer kernels and cuBLAS; its native
NVFP4 MMA path is compiled in and unused because no NVFP4 GGUF is in the store.

## Measurement: `x/mlxrunner/bench/qqmm` (PR #286)

A standalone tool on the fork's MLX binding times, per layer shape of gemma4:31b and
qwen3.8:27b (dims from the store's `config.json`) and per row count M, four ways to run the
layer: `bf16` (cuBLASLt on unquantised weights, the reference), `qmm` (today), `qqmm`
(cuBLASLt FP4), and `dequant` (the nvfp4 weights dequantised to bf16 on every call, then the
bf16 GEMM). Each cell: median / p10 / p90 / first-call wall ms over 20 calls, TFLOP/s, and the
error against an fp32 product with the unquantised weights. Records:
`vision-suite/preflight/runs/qqmm-blackwell-contended-2026-09-08.jsonl`,
`qqmm-2080ti-2026-09-08.jsonl`; logs on the CUDA host in `preflight-runs/qqmm-*.log`.

**Conditions.** The Blackwell run was beside the MLX think-on campaign (GPU at 94 %, host load
20): time-slicing puts a floor of about 2.3–3.0 ms under every small kernel, so **only the
M ≥ 2048 rows carry signal** and even those are inflated; the M ≤ 512 rows and every M=1 cell
are contention, not kernels. A quiet-GPU run (acceptance criterion 1) replaces these numbers;
the direction does not depend on it. The 2080 Ti was idle.

### RTX PRO 6000 Blackwell (cc 12.0), beside the campaign — the four MLP projections

#### gemma4-31b gate_proj (K=5376, N=21504)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | qqmm ms (TFLOP/s) | dequant ms (TFLOP/s) | qmm / bf16 | qmm / qqmm | qmm / dequant | relRMS bf16 / qmm / qqmm / dequant |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.612 (0) | 4.862 (0) | 3.054 (0) | 2.926 (0) | 1.86x | 1.59x | 1.66x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 8 | 2.640 (1) | 2.755 (1) | 2.265 (1) | 5.110 (0) | 1.04x | 1.22x | 0.54x | 3.1e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 64 | 2.604 (6) | 2.599 (6) | 4.680 (3) | 5.210 (3) | 1.00x | 0.56x | 0.50x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 512 | 3.073 (39) | 4.810 (25) | 4.699 (25) | 4.797 (25) | 1.57x | 1.02x | 1.00x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 2048 | 3.865 (123) | 10.821 (44) | 5.030 (94) | 5.868 (81) | 2.80x | 2.15x | 1.84x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 4096 | 12.008 (79) | 15.669 (60) | 3.364 (282) | 11.428 (83) | 1.30x | 4.66x | 1.37x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |

#### gemma4-31b down_proj (K=21504, N=5376)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | qqmm ms (TFLOP/s) | dequant ms (TFLOP/s) | qmm / bf16 | qmm / qqmm | qmm / dequant | relRMS bf16 / qmm / qqmm / dequant |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.716 (0) | 0.071 (3) | 2.596 (0) | 3.109 (0) | 0.03x | 0.03x | 0.02x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 8 | 2.671 (1) | 3.169 (1) | 2.747 (1) | 3.143 (1) | 1.19x | 1.15x | 1.01x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 64 | 2.702 (5) | 3.094 (5) | 2.365 (6) | 3.094 (5) | 1.15x | 1.31x | 1.00x | 3.4e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 512 | 2.586 (46) | 2.941 (40) | 2.857 (41) | 3.574 (33) | 1.14x | 1.03x | 0.82x | 3.5e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 2048 | 4.048 (117) | 8.655 (55) | 3.106 (152) | 5.029 (94) | 2.14x | 2.79x | 1.72x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 4096 | 7.810 (121) | 16.559 (57) | 3.766 (251) | 8.610 (110) | 2.12x | 4.40x | 1.92x | 3.3e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |

#### qwen3.8-27b gate_proj (K=5120, N=17408)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | qqmm ms (TFLOP/s) | dequant ms (TFLOP/s) | qmm / bf16 | qmm / qqmm | qmm / dequant | relRMS bf16 / qmm / qqmm / dequant |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.558 (0) | 0.059 (3) | 2.816 (0) | 2.910 (0) | 0.02x | 0.02x | 0.02x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 8 | 2.509 (1) | 2.922 (0) | 2.450 (1) | 3.125 (0) | 1.16x | 1.19x | 0.94x | 3.1e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 64 | 2.920 (4) | 2.356 (5) | 2.547 (4) | 3.109 (4) | 0.81x | 0.93x | 0.76x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 512 | 2.673 (34) | 3.287 (28) | 1.189 (77) | 3.417 (27) | 1.23x | 2.76x | 0.96x | 3.1e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 2048 | 3.628 (101) | 7.682 (48) | 2.839 (129) | 3.738 (98) | 2.12x | 2.71x | 2.05x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 4096 | 4.627 (158) | 11.564 (63) | 3.212 (227) | 7.453 (98) | 2.50x | 3.60x | 1.55x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |

#### qwen3.8-27b down_proj (K=17408, N=5120)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | qqmm ms (TFLOP/s) | dequant ms (TFLOP/s) | qmm / bf16 | qmm / qqmm | qmm / dequant | relRMS bf16 / qmm / qqmm / dequant |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.357 (0) | 0.100 (2) | 0.176 (1) | 0.548 (0) | 0.28x | 0.57x | 0.18x | 2.9e-03 / 1.1e-01 / 1.4e-01 / 1.1e-01 |
| 8 | 2.473 (1) | 3.495 (0) | 2.839 (1) | 4.042 (0) | 1.41x | 1.23x | 0.86x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 64 | 2.517 (5) | 3.309 (3) | 3.637 (3) | 3.908 (3) | 1.31x | 0.91x | 0.85x | 5.4e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 512 | 3.578 (26) | 0.727 (126) | 0.464 (197) | 0.993 (92) | 0.20x | 1.57x | 0.73x | 4.1e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 2048 | 3.660 (100) | 7.215 (51) | 3.036 (120) | 4.387 (83) | 1.97x | 2.38x | 1.64x | 2.9e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |
| 4096 | 4.469 (163) | 12.104 (60) | 3.225 (226) | 7.473 (98) | 2.71x | 3.75x | 1.62x | 3.1e-03 / 1.0e-01 / 1.4e-01 / 1.0e-01 |

### RTX 2080 Ti (cc 7.5), idle — one shape, the others say the same

#### gemma4-31b gate_proj (K=5376, N=21504)

| M | bf16 ms (TFLOP/s) | qmm ms (TFLOP/s) | qqmm ms (TFLOP/s) | qqmm / qmm | relRMS bf16 / qmm / qqmm |
|---|---|---|---|---|---|
| 1 | 0.459 (1) | 0.952 (0) | 0.830 (0) | 1.15x | 2.9e-03 / 1.0e-01 / 1.4e-01 |
| 8 | 2.061 (1) | 8.433 (0) | 8.507 (0) | 0.99x | 2.9e-03 / 1.0e-01 / 1.4e-01 |
| 64 | 2.137 (7) | 22.464 (1) | 22.486 (1) | 1.00x | 2.9e-03 / 1.0e-01 / 1.4e-01 |
| 512 | 16.698 (7) | 145.428 (1) | 146.641 (1) | 0.99x | 2.9e-03 / 1.0e-01 / 1.4e-01 |
| 2048 | 65.943 (7) | 620.483 (1) | 619.697 (1) | 1.00x | 2.9e-03 / 1.0e-01 / 1.4e-01 |
| 4096 | 134.901 (7) | 1278.075 (1) | 1268.947 (1) | 1.01x | 2.9e-03 / 1.0e-01 / 1.4e-01 |

### Two runs after the campaign (2026-09-08 17:14 and 17:17; production still at 79 % GPU)

`qqmm-blackwell-quiet-{1,2}-2026-09-08.jsonl`. "Quiet" on this host means production only; the
time-slicing floor still shows in the M ≤ 512 cells, so the reading stays on M ≥ 2048. Median ms
per call, run 1 / run 2, and the speed-up of each path over `qmm` (run 1 / run 2):

| projection | M | bf16 | qmm | dequant | qqmm | dequant vs qmm | qqmm vs qmm |
|---|---|---|---|---|---|---|---|
| gemma4:31b gate | 2048 | 3.98 / 4.05 | 6.33 / 7.77 | 4.25 / 4.20 | 2.30 / 2.68 | 1.5× / 1.9× | 2.8× / 2.9× |
| gemma4:31b gate | 4096 | 5.92 / 7.73 | 11.85 / 13.21 | 7.98 / 7.39 | 3.05 / 3.02 | 1.5× / 1.8× | 3.9× / 4.4× |
| gemma4:31b down | 2048 | 3.79 / 3.95 | 8.42 / 8.26 | 4.20 / 4.28 | 2.97 / 2.83 | 2.0× / 1.9× | 2.8× / 2.9× |
| gemma4:31b down | 4096 | 7.72 / 7.80 | 15.19 / 15.83 | 8.31 / 8.17 | 3.59 / 3.44 | 1.8× / 1.9× | 4.2× / 4.6× |
| qwen3.8 gate | 2048 | 3.55 / 3.36 | 6.76 / 7.34 | 3.74 / 3.80 | 2.93 / 2.62 | 1.8× / 1.9× | 2.3× / 2.8× |
| qwen3.8 gate | 4096 | 4.72 / 4.64 | 10.98 / 12.63 | 7.16 / 7.26 | 3.08 / 3.31 | 1.5× / 1.7× | 3.6× / 3.8× |
| qwen3.8 down | 2048 | 3.37 / 3.43 | 7.28 / 7.30 | 3.93 / 1.55 | 2.87 / 0.60 | 1.9× / 4.7× | 2.5× / 12× |
| qwen3.8 down | 4096 | 4.58 / 4.90 | 12.12 / 11.22 | 7.48 / 7.29 | 3.34 / 3.31 | 1.6× / 1.5× | 3.6× / 3.4× |

(Derived from the two runs' JSONL; the runs are the record. The run-2 qwen3.8 down 2048 cell
caught a quiet window, which is what the floor does in the other direction.) Repeatable at
M ≥ 2048: the dense path is 1.5–2.0× `qmm`, the FP4 GEMM 2.3–4.6×, on every MLP projection.
At M=1 the dense path costs 0.7–3 ms against 0.06–0.2 ms for `qmm` — the copy — so the row
threshold is not optional. The crossover lies between 512 and 2048 rows and is not resolved by
these runs (no 1024 point); criterion 3's one-image shape (1122 rows) probes it directly.

### Reading

- **Today's kernel is slower than plain bf16 at prefill sizes.** At M=4096 `qmm` reaches
  55–73 TFLOP/s on the MLP projections against 100–160 for bf16 cuBLASLt: the in-register
  dequant costs more than the 4-bit weights save once M is large.
- **The dense path with a per-call dequant recovers most of that**: 1.4–2.1× over `qmm` at
  M ≥ 2048 on the MLP projections, with the error identical to `qmm` in every cell (same
  weights, bf16 math). The transient copy is the price; below ~512 rows it is slower than
  `qmm`, so the policy is a row threshold.
- **The FP4 GEMM is real on this card**: 3.6–4.7× over `qmm` and about 2× over bf16 at
  M=4096, at the cost of quantising the activations: relative RMS 0.14 against 0.10 on
  Gaussian operands, and real activations have outliers. That is layer 2 below, behind an
  accuracy gate.
- **Turing gets nothing**: `qqmm` falls back to the `qmm` kernels below cc 10 while still
  quantising the activations, and `qmm` itself is MLX's naive kernel there at ~0.8 TFLOP/s,
  nine times slower than bf16 on the same card. A 2080 Ti is unfit for MLX nvfp4 in any form.

## Sizing: how often prefill is hit, and what the opt-in saves

Prefill runs once per request over the whole prompt on the teacher-v3 loop (fresh single-image
requests; the prefix cache can reuse only a shared prefix) and over the new suffix only in a
chat. The runner prefills in 2048-token chunks, and the dense path triggers per matmul call
when the chunk carries at least the threshold rows: with the threshold at 1024, a 3.5k-token
prompt hits it on both chunks (2048 + 1483), a 1.2k-token prompt (one gemma4 image plus text)
once, prompts under 1024 tokens never. Decode never.

**The production shape** (`:11497`, 245 requests in the log tail of 2026-09-08, the GGUF path):

| | p50 | p90 | max |
|---|---|---|---|
| prompt tokens | 3531 | 3908 | 4799 |
| generated tokens | 2136 | 3255 | — |
| prefill time | 1.83 s | 2.08 s | — |
| generation time | 45.5 s | 69.2 s | — |
| prefill share of the request | 4 % | 5 % | — |

Over the tail: 444 s of prefill against 12,130 s of generation, 3.5 %.

**The same shape on the MLX path** (T1 throughput cells of the `main` think-off validation,
2026-09-06; prefill tok/s there includes the vision tower):

| model | prefill 3.5k tokens | generate 2.1k tokens | prefill share | opt-in saves per request |
|---|---|---|---|---|
| gemma4:31b-nvfp4 | 5.2 s (681 tok/s) | 93 s (23 tok/s) | 5 % | ~1–2 s, about 1.5 % |
| gemma4:26b-nvfp4 | 3.2 s (1116) | 67 s (32) | 4.5 % | ~1 s, about 1.5 % |
| gemma4:12b-nvfp4 | 3.5 s (1016) | 69 s (31) | 4.8 % | ~1 s, about 1.5 % |
| qwen3.8:27b-nvfp4 | 0.7 s (5222) | 134 s (16) | 0.5 % | ~0.2 s, about 0.2 % |
| qwen3.6:35b-a3b-nvfp4 | 0.7 s (4908) | 112 s (19) | 0.6 % | ~0.2 s, about 0.2 % |

The "saves" column assumes the GEMMs are 60–70 % of prefill time and take the measured
1.6–2× at 2048 rows; the rest of prefill (attention, the vision tower, norms) is untouched.
**For the production shape the opt-in is worth one to two percent on gemma4 and nothing
measurable on qwen.** It pays where prefill dominates:

- **long text prompts on qwen3.8**: 41.6k tokens prefill in 31 s on `main` (slope probe,
  2026-09-06) and that model's prefill is matmul-heavy — up to a third off, ~10 s per request;
- **gemma4 long prompts are attention-bound** (the prefill transient task): little gain;
- **many-image prompts** (three images ≈ 3.3k tokens) sit at the production shape: 1–2 %;
- **batched prefill** (`OLLAMA_NUM_PARALLEL` > 1) multiplies the rows per call and moves every
  chunk past the threshold.

So: a correct, free-of-accuracy-risk speed-up that is small for today's workload and meaningful
for long-prompt and batched use. Off by default; measured before it is turned on anywhere.

## Design (landed)

- `nn.PrefillDequantRows` (env `OLLAMA_MLX_PREFILL_DEQUANT_ROWS`, default 0 = off). The policy
  `useDequantGEMM(rows, threshold, mode, cuda)`: opted in, `nvfp4` only (the measured mode),
  CUDA only (Metal's kernels were not measured), rows ≥ threshold. `rows` is everything but
  the last dim of the activation.
- `QuantizedLinear.denseGEMM`: `mlx.Dequantize` of the layer to the activation dtype, then the
  dense matmul against the transposed view. The global-scale and bias handling after the matmul
  is shared with today's path. The dequantised copy is a graph temporary: ~230 MB for a 31b MLP
  projection, one layer at a time, nothing resident.
- Not bit-identical to `qmm` (accumulation order), hence criterion 2. Unit tests: the env
  parser, the policy table, and the branch body against the mixed-input kernel on the same
  nvfp4 weights (runs wherever MLX runs).

## Runner measurements, 2026-09-08 (flag off vs on, same binary, same container shape)

`pr287-4bf13c65` (this branch's Go binary swapped onto `main-a523d60b`; native payload unchanged),
`OLLAMA_MLX_PREFILL_DEQUANT_ROWS=1024`, explicit `num_ctx` 65536, `num_predict` 32, one container
per flag state, beside production. Records: `preflight/runs/prefill-dequant-text-2026-09-08.jsonl`
(long text) and `prefill-dequant-images-2026-09-08.jsonl` (images; a nonce in a **system** message
defeats the prefix cache — the `[img-N]` tags precede the user content, so a nonce inside the
content does not, and a first attempt measured cache hits). Peaks are the runner's own
`peak memory` line, which `TextGenerationPipeline` resets per request, compared at equal position
in an identical request sequence.

| shape | rows | prefill off → on | speed-up | peak off → on |
|---|---|---|---|---|
| gemma4:31b, 1 image | 1134 | 418 → 896 tok/s | **2.13×** | 30.7 → **37.5** GiB (+6.8) |
| gemma4:31b, 3 images | 3340 | 376 → 603 tok/s | **1.60×** | 36.7 → **40.4** GiB (+3.8) |
| qwen3.8, 1 image | 2338 | 961 → 1210 tok/s | 1.26× | 32.9 → 34.1 GiB (+1.2) |
| qwen3.8, 3 images | 3064 | 1006 → 1162 tok/s | 1.16× | 34.9 → 35.7 GiB (+0.8) |
| gemma4:31b, text 41.6k | 41644 | 82 → 96 tok/s | 1.17× | 53.7 → 53.7 GiB (0.0) |
| qwen3.8, text 41.6k | 41642 | 1130 → 1550 tok/s | 1.37× | 49.0 → 49.2 GiB (+0.2) |

**The speed-up is real and larger on images than the microbenchmark suggested** (a vision prefill
is almost all matmul over the image tokens). **The memory cost is real too, and it was not
predicted:** up to +6.8 GiB on gemma4:31b.

### Why the transient is layers wide, not one layer

The PR's first claim — "one transient copy of one layer, ~230 MB" — is **wrong for a lazily
evaluated graph**. MLX builds the whole chunk's forward pass before evaluating it, so the
dequantised bf16 copies of many layers are live simultaneously; on 31b the MLP pair alone is
~460 MB per layer over 60 layers. The long-text shape hides it because its peak is set by the
attention transient at a different moment, which is why the text rows show no change.

**This blocks enabling the flag anywhere, even opt-in, on a shared card.** Admission (#276)
prices weights + KV + a per-architecture headroom; several GiB of unpriced prefill transient on a
~40 GiB budget is precisely the failure mode
[the prefill-transient task](mlx-prefill-transient-scales-with-context.md) documents. Bound the
copies first — force evaluation per layer, or dequantise into one reused buffer — then re-measure
peak before the threshold default is set.

## Rework attempt 1 (2026-09-10): release the handle — **did not bound the transient**

`mlx.Release` frees a handle the caller exclusively owns; `denseGEMM` calls it on the
dequantised copy as soon as the matmul node holds it, on the theory that a live Go handle was
retaining the buffer through the chunk's eval (the prefill loop's own comment says a live handle
does exactly that, `pipeline.go`). Measured on `pr287b-5c9705e7`, same probe, same host
(`preflight/runs/prefill-dequant-images-reworked-2026-09-10.jsonl`):

| model, shape | peak off | peak on, before | peak on, released | Δ vs off | speed-up |
|---|---|---|---|---|---|
| gemma4:31b, 1 image | 30.7 | 37.5 | **37.7** | +7.0 | 2.33× |
| gemma4:31b, 3 images | 36.7 | 40.4 | **40.4** | +3.8 | 1.84× |
| qwen3.8, 1 image | 32.9 | 34.1 | **33.8** | +0.8 | 1.39× |
| qwen3.8, 3 images | 34.9 | 35.7 | **35.7** | +0.8 | 1.19× |

The flag-off control reproduced the earlier run to 0.1 GiB on all four shapes, so the probe is
repeatable and the untouched path is unchanged. **The transient is unchanged**, so the retention
is not the Go handle: whatever holds those buffers alive is inside MLX's own graph or allocator.
`mlx.Release` is kept — it is correct, tested, and cheap — but it is not the fix, and this
disproves the mechanism stated in the previous section.

### Rework attempt 2 (same day): force the layer to evaluate — **strictly worse, it OOMs**

`mlx.Eval(out)` inside `denseGEMM`, so a dequantised copy cannot outlive its own layer
(`preflight/runs/prefill-dequant-images-perlayer-eval-2026-09-10.jsonl`). On gemma4:31b's very
first one-image request the peak went to **64.7 GiB and the request died** with
`cudaMallocAsync … out of memory`, against 37.7 GiB for the same shape without the Eval. The
flag-off control in the same run was unchanged (30.7 GiB), so it is the eager evaluation that
costs the memory. Stopped there.

**Reading: the lazy graph was helping, not hurting.** Evaluating layer by layer defeats MLX's own
scheduling — it materialises each layer's copy at a point of its choosing and prevents whatever
reuse the whole-graph evaluation was doing — so the transient is neither the Go handles (attempt
1) nor a graph-lifetime artefact that eager evaluation can bound (attempt 2).

### Where that leaves the approach

**llama.cpp's fix does not port.** Its bound comes from ggml evaluating node by node into a
**reused pool buffer** it owns; MLX's Go API has no out-parameter matmul, no donation control and
no scratch arena at this level, so there is no way to express "dequantise into the same buffer
each layer" from here. Bounding it would need an MLX-side change (a scratch/donation API, or a
fused dequantise-matmul that never materialises the copy).

Two honest options remain, and neither is this PR as written:

1. **Price it instead of bounding it.** The transient is measurable and per-architecture (+7.0 GiB
   gemma4:31b, +0.8 qwen3.8). `admissionHeadroom` could add it when the flag is set, which turns
   an unpriced risk into a priced one exactly as #276 does for everything else. Cheap, honest, and
   it makes the flag safe on a shared card — but it *buys* the speed-up with budget, so on a card
   that is already tight it will refuse loads that succeed today.
2. **Take it upstream.** A fused nvfp4 dequantise-matmul, or a scratch-buffer API, is the real
   fix and belongs in MLX; the microbenchmark in #286 is the evidence to open that conversation
   with. Note MLX's mixed-input kernel is what loses here — llama.cpp's equivalent (MMQ) *wins*
   against dequantise+cuBLAS on tensor-core hardware, so the gap is a kernel-quality gap.

Criterion 2 (the path changes output deterministically) is unaffected by any of this and remains
a separate blocker.

## Ceiling probe (2026-09-10): a Triton mixed-input kernel is **slower** than MLX's

The "make the streaming kernel fast" option needs a number: what can a well-tiled kernel that
unpacks 4-bit weights in registers and multiplies against bf16 activations actually reach on
sm120? `kernels/triton_mixed_input_gemm.py` implements one (Triton 3.3.1, torch 2.7.1+cu128) and
benchmarks it against bf16 cuBLAS on the same dequantised weights, autotuning 24 tile/stage
configurations per shape. Log: `preflight/runs/triton-mixed-input-2026-09-10.log`.

| shape | M | Triton mixed-input | bf16 cuBLAS | ratio | rel err |
|---|---|---|---|---|---|
| gemma4-31b gate | 2048 | 38.6 TF/s | 117.0 | 0.33× | 3.2e-03 |
| gemma4-31b gate | 4096 | 48.0 | 207.4 | 0.23× | 3.2e-03 |
| gemma4-31b down | 2048 | 53.9 | 197.6 | 0.27× | 3.2e-03 |
| gemma4-31b down | 4096 | 41.4 | 171.3 | 0.24× | 3.2e-03 |
| qwen3.8 gate | 2048 | 51.2 | 200.6 | 0.26× | 3.2e-03 |
| qwen3.8 gate | 4096 | 53.2 | 219.1 | 0.24× | 3.2e-03 |
| qwen3.8 down | 2048 | 52.0 | 191.8 | 0.27× | 3.2e-03 |
| qwen3.8 down | 4096 | 51.1 | 187.9 | 0.27× | 3.2e-03 |

**Rerun pending:** the GPU was shared with production throughout. Re-run
`kernels/triton_mixed_input_gemm.py` on a quiet Blackwell before quoting these as final; the
ratios should hold but the absolute figures will move.

**Is it the CUDA 12.8 toolchain?** (Glenn, 2026-09-10: 12.8 was the first release with sm_120.)
It splits by component:

| component | toolchain that compiled it | could 12.8 explain it? |
|---|---|---|
| MLX, production (the slow kernel) | CUDA **13.0** (NVRTC 13.0.88, cudart 13.0.96, cuBLAS 13.1.1.3), JIT flag `--gpu-architecture=sm_120a` from `jit_module.cpp` | **no** — not 12.8, and it targets sm_120a natively; the Ampere-era `SM80_16x8x16` MMA and `cp.async` are chosen in MLX's *source*, which no toolkit version changes |
| this Triton prototype | Triton 3.3.1's bundled `ptxas-blackwell`, **CUDA 12.8.61** | **plausibly** — early sm_120 code generation could schedule worse |
| bf16 cuBLAS reference | torch 2.7.1+cu128, cuBLAS 12.8 | no — it reaches 117–219 TF/s on the same card, so 12.8 drives sm_120 tensor cores well |

The Triton half is testable with one variable: the host has `/usr/local/cuda-13.0/bin/ptxas`, and
Triton reads `TRITON_PTXAS-BLACKWELL_PATH` (a hyphenated name, so it has to go through `env`, not
`export`). `claude-scratch/ab-triton-ptxas.sh` runs the prototype with both, in separate Triton
caches so each recompiles; it is staged for the quiet-GPU rerun, not run.

The error is bf16 rounding against the dequantised reference, i.e. the kernel is correct.

**It reaches 23–33 % of cuBLAS, which is *below* MLX's own kernel** (55–73 TF/s at 4096 rows in
the #286 harness, roughly 40–60 % of cuBLAS measured there). **So this does not establish the
ceiling — it refutes the cheap hypothesis** that MLX's kernel is simply badly written and a
rewrite walks to cuBLAS parity. It is a competent kernel; beating it takes Marlin-class work.

Caveats, so nobody over-reads the table: the absolute cuBLAS figures here (117–219) come from
`do_bench` medians on a warm GPU and are not comparable with the wall-clock-per-call figures in
the #286 tables (100–160) — compare ratios within a harness, not numbers across harnesses. And
this prototype is deliberately simple: it splits K in half to keep both nibble halves contiguous,
which doubles the number of `tl.dot` calls at half the K each; it reloads the per-16 scales 8×
redundantly instead of staging them in shared memory; and it has no swizzling or hand-tuned
pipelining beyond `num_stages`. A serious attempt would pre-shuffle the weight layout the way
Marlin does.

**What this changes.** Mixed-input tops out at bf16-dense speed *by construction* — the 4-bit
tensor cores need both operands in 4-bit — so even a perfect kernel here buys at most ~2× over
MLX today, for days of kernel work, on the one shape class that is matmul-bound. The 4-bit path
already exists for this chip (CUTLASS ships `sm120_blockscaled_mma_tma` collectives and worked
nvfp4 examples) and is worth 2–4×, but its blocker is the accuracy question, not kernels. **The
kernel work is not where the leverage is.**

## What the Metal path does differently, and what is worth borrowing

Glenn's observation (2026-09-10): MLX-CUDA nvfp4 is slower than GGML q4_K_M, yet on MLX-Metal
nvfp4 is roughly on par with q4_K_M. **We cannot verify the Metal half here** — this host has no
Metal campaigns and the two platforms are never mixed — so treat the premise as his, not as
measured. The architectural difference behind it is real and checkable in the MLX source:

| | Metal | CUDA |
|---|---|---|
| dense GEMM | MLX's own `steel` template (`BlockMMA`, `BlockLoader`), tuned by them | **cuBLAS / cuBLASLt** — NVIDIA's, not extensible |
| quantised GEMM | **the same `steel` template with a `QuantizedBlockLoader` swapped in for the weight operand** (`metal/kernels/quantized.h:1228-1271`) | a **standalone** CuTe kernel with its own tiled MMA and `cp_async` pipeline (`device/qmm_sm80.cuh`) |
| consequence | quantised inherits every dense tuning; dequantisation is fused into the tile load | quantised inherits nothing; it is a separate artefact competing against decades-tuned kernels |

**That is the asymmetry, and it is not about the format.** On Metal the comparison is MLX-vs-MLX:
the quantised kernel *is* the dense kernel plus a loader, so it keeps pace by construction. On
CUDA the comparison is MLX's own kernel against ggml's MMQ, hand-tuned over years — a kernel
maturity gap. Our Triton probe (above) supports that reading: a competent from-scratch mixed-input
kernel landed *below* MLX's, so the deficit is not carelessness, it is the absence of a tuned
template to build on.

**The borrowable concept: make the quantised CUDA path a loader swap on a tuned GEMM template,
not a bespoke kernel.** CUDA's equivalent of `steel` is a CUTLASS collective mainloop, and
CUTLASS's Hopper mixed-dtype example is exactly that shape — standard collective plus a custom
operand converter — so an sm120 version would inherit TMA, warp specialisation and the tile
tuning instead of re-deriving them. A second, cheaper borrow: Metal parameterises **one** loader
by quantisation mode, where CUDA carries a zoo of kernels behind capability gates
(`qmm_sm90` pinned to cc 9, `qmm_sm80`, `qmm_naive`, `qmv`, `fp_qmv`), which is how a Blackwell
card ends up on an Ampere kernel.

**Caveat on transferability.** Blackwell has far more tensor-core throughput per byte of
bandwidth than an Apple GPU, so the ALU spent unpacking 4-bit weights is proportionally more
exposed on CUDA: parity-with-dense is a harder target there than on Metal, even with the same
architecture. This is reasoning from the hardware balance, not a measurement.

**This is the well-specified version of the upstream ask.** "Make the streaming kernel fast" is
not actionable; "build the CUDA quantised path on a CUTLASS collective with a dequantising
loader, as the Metal path does on steel" is.

## Similar reports elsewhere (searched 2026-09-10)

Others on consumer Blackwell hit the same wall from the other side:

- **vLLM on RTX 5090 / RTX PRO 6000** (vllm-project/vllm#47749, #48199): weight-only NVFP4
  checkpoints (`W4A16_NVFP4`) are pinned to Marlin, which dequantises weights to the activation
  dtype inside the kernel, because those checkpoints carry no activation scales and *native FP4
  GEMM needs FP4 activations*. That is the structural finding above, stated by vLLM's own code.
- **vLLM #55405**: on sm_12x a W4A4 checkpoint was silently routed to a dequantise-to-16-bit
  kernel, costing **~31 % prefill on GB10**, recovered by forcing the native W4A4 kernel — a
  measured price of mixed input against native FP4 on consumer Blackwell.
- **CUTLASS #3096, on our exact card** (4× RTX PRO 6000): native NVFP4 MoE grouped GEMM produced
  garbage; the working fix combined FlashInfer sm120 patches with **`compute_120f` built by
  CUDA 13.0**, and even then native FP4 ran **14.6 tok/s against Marlin W4A16's 46–49** because
  the fast warp-specialised TMA tactics failed to initialise. On this card a mature mixed-input
  kernel beat an immature native one — the kernel-maturity reading again.
- **MLX #4339, GB10 (sm_121)**: `gather_qmm` ~24× slower than equal-FLOP `quantized_matmul`;
  quantised MoE prefill 6.5× slower than the same model in bf16. Our qwen3.6 and gemma4:26b are
  MoE and take that path. Dense quantised reached ~85 % of bf16 there against our 40–60 %,
  consistent with GB10 being bandwidth-poor, where 4-bit weights win more — the Metal argument.

## Toolchain facts, checked with this host's own `ptxas` (2026-09-10)

| probe | ptxas 12.8 | ptxas 13.0 |
|---|---|---|
| `wgmma` on sm_90a | assembles | assembles |
| `wgmma` on **sm_120a** | **not supported** | **not supported** |
| `tcgen05` on sm_100a | assembles | assembles |
| `tcgen05` on **sm_120a** | **not supported** | **not supported** |
| family target `sm_120f` | **not defined** | assembles |

- **No toolkit version unlocks Hopper or datacenter-Blackwell instructions on this card**, so
  "widen MLX's sm90 gate to cc 12" is impossible, not merely untested. A community wiki claiming
  `wgmma` runs on sm_120 at lower throughput is wrong.
- **The family target is the one genuinely version-gated item**: `sm_120f` does not exist in 12.8
  and does in 13.0, and CUTLASS #3096's working fix needed it. MLX targets `sm_120a`, which 12.8
  has, and its prefill kernel uses no family features — so this does not explain MLX's speed.
- **Driver-reported limits** (`torch.cuda.get_device_properties`): 99 KiB opt-in shared memory per
  block, 100 KiB per SM, 65,536 registers per SM, 188 SMs, 128 MiB L2. The Hopper and
  datacenter-Blackwell designs assume far more shared memory, so any sm120 kernel must size its
  tiles and pipeline depth to ~100 KiB — a real reason MLX's tiles are small.

## A real sm120 mixed-input kernel (2026-09-11) — **the kernel route is viable after all**

A Fable 5.1 agent took the loader-swap question to CUTLASS. Code on branch
`feat/sm120-mixed-input-gemm` (not merged, no PR), under `docs/maxusai/kernels/sm120_mixed_input/`;
logs `claude-scratch/wt-sm120mi-*.log`. Claims below were spot-checked against CUTLASS and the logs.

**Composition through CUTLASS's builders is impossible**, as the interim check found: the sm120
builder accepts only 4/6/8-bit operands, the atom selector returns only the narrow-format
`SM120_16x8x32` shape, and the mainloop's operand-transform slots are never invoked. **But the
tuned mainloop underneath exists and is generic over the MMA atom.** Hand-instantiating
`MainloopSm120TmaWarpSpecialized` with the Ampere bf16 atom, bypassing `CollectiveBuilder`,
compiled first time and runs at 0.8–1.2× cuBLAS. The Metal analogy does transfer, once the
builder is bypassed.

**The kernel:** a ~580-line derivative of CUTLASS's `sm120_mma_tma.hpp` that keeps its TMA
producer/`mma.sync` consumer split and pipeline, adds TMA streams for packed weights and scales,
and dequantises e2m1→bf16 in registers between fragment load and MMA — the slot the dense loop
uses for `ldmatrix`. Weights use a Marlin-style fragment-native layout, a pure permutation of
MLX's native layout. Numerically exact by construction (e2m1 × e4m3 fits bf16's mantissa);
measured max error 2.6–3.3e-3, zero elements over 5 % — the error profile of cuBLAS bf16.

**Measured, interleaved in one process, `do_bench` median of 3 rounds, GPU shared throughout**
(`wt-sm120mi-ab-v2v3v4.log`, best version v3 with bf16 scales; M=2048 rows were stable to <1 %):

| shape, M=2048 | v3 | cuBLAS bf16 | ratio |
|---|---|---|---|
| gemma4:31b gate | 238 TF/s | 285 | **0.84** |
| gemma4:31b down | 229 | 290 | **0.79** |
| qwen3.8 gate | 232 | 285 | **0.81** |
| qwen3.8 down | 217 | 336 | **0.65** |

The M=4096 rows swung up to 2× between rounds, dense kernel included; re-measure on a quiet GPU.

**Against MLX's kernel, stated carefully.** The agent's "~3× MLX" compares absolute TFLOP/s across
two harnesses whose cuBLAS readings differ ~2× (its `do_bench` medians against the #286 wall-clock
bench), so it overstates. Comparing each kernel's ratio to cuBLAS *within its own harness* —
MLX `qmm` reached 0.45–0.63 of bf16 at M=2048 in the quiet #286 runs — gives **~1.3–1.8×
over MLX's kernel**. Veto: a same-harness run of MLX's `quantized_matmul`, not yet done. (The
Triton table above also used `do_bench`'s default *mean* under contention, so its absolutes are
pessimistic; its in-run ratios stand.)

**Where the remaining gap is**, measured with probe builds (`wt-sm120mi-probes.log`): with the
e2m1→bf16 conversion compiled out but every load, TMA stream and stage intact, the kernel runs
at **0.96–1.03× cuBLAS**. So the loader/pipeline swap costs nothing; **the whole overhead is the
conversion arithmetic on the MMA warps**, which this chip's tensor pipe does not hide (~12–15
non-MMA instructions per HMMA in the SASS).

**Next step, the agent's estimate 2–3 days:** move dequantisation off the MMA warps into
producer/transform warps that write bf16 B tiles to shared memory, so the consumers run the dense
loop unmodified — the sm100 transform-warpgroup pattern. On sm120 three of the four producer
warps idle today. The constraint is shared memory: a bf16 B tile is 16 KB per stage against a
99 KB budget. The probe result is the evidence it would land near dense speed.

**What this changes in the conclusions above.** "Kernel work is not where the leverage is" was
drawn from a naive Triton prototype at 0.25× cuBLAS; a CUTLASS-mainloop derivative reaches 0.65–
0.84×, and the structural cap (mixed input tops out at bf16-dense speed) still holds but is now
within reach rather than theoretical. Unlike the dequant-to-bf16 opt-in in this PR, this path
needs **no full-size weight copy**, so the +6.8 GiB failure does not apply. **The deterministic
output change does apply** (corrected 2026-09-11; an earlier version of this paragraph said
otherwise): any kernel that accumulates in a different order from MLX's `qmm` changes model output
bit-for-bit, exactly as the `document_single` control showed for this PR's path, so it has to be
judged against the vision suite the same way, as a numerics change. Prototype limits: N % 128 = 0, K % 64 = 0, TN, batch 1,
synthetic weights, not integrated into MLX.

## The producer-warp version (2026-09-11) — 0.92–0.95× the dense kernel

A second agent moved the e2m1→bf16 conversion off the eight MMA warps onto producer warps that
were idle, writing bf16 B tiles to shared memory so the MMA warps run the dense loop unmodified.
Branch `feat/sm120-mixed-input-gemm` head `946dd83a` (4 commits on `bfa3da0e`, no PR), files
`sm120_mma_tma_transform.hpp` and `sm120_gemm_tma_ws_cooperative_transform.hpp`; selectable
beside v3 by `-DNVFP4_TRANSFORM_WARPS=T`. Verified: repository state, two table rows against
`claude-scratch/wt-sm120mi-ab-final3.log` to the decimal, and zero elements >5 % off across all
40 validations in that log.

**Numerically exact** on every build: max|err|/max|ref| 2.5–3.7e-3, the cuBLAS bf16 profile.
**No weight repack at all**: it reads MLX's native `uint32 [N][K/8]` words; only the scales need
a host permutation, to bf16 in a row-pair interleave, because a TMA box row must be ≥16 B.

Interleaved, `do_bench` median of 3 rounds, GPU shared at 66–100 % foreign utilisation, TFLOP/s:

| shape | M | tw2s3 | v3 | dense sm120 | cuBLAS | tw2s3/dense | tw2s3/cuBLAS |
|---|---|---|---|---|---|---|---|
| gemma4-31b gate | 2048 | 264 | 243 | 282 | 287 | 0.93 | 0.92 |
| gemma4-31b gate | 4096 | 266 | 246 | 289 | 325 | 0.92 | 0.82 |
| gemma4-31b down | 2048 | 260 | 232 | 272 | 290 | 0.95 | 0.90 |
| gemma4-31b down | 4096 | 246 | 221 | 262 | 290 | 0.94 | 0.85 |
| qwen3.8 gate | 2048 | 249 | 231 | 328 | 282 | 0.76 | 0.88 |
| qwen3.8 gate | 4096 | 271 | 248 | 294 | 306 | 0.92 | 0.88 |
| qwen3.8 down | 2048 | 238 | 216 | 259 | 337 | 0.92 | 0.71 |
| qwen3.8 down | 4096 | 263 | 231 | 282 | 288 | 0.93 | 0.91 |

Every row beats v3, by 8–14 %. The 0.76 row is against an outlier dense round. The gemma M=2048
rows are **bimodal for the transform variants only** — the same binary at 1.50 or 1.80 ms across
rounds while v3, cuBLAS and dense stay within 1 % — unattributed; treat those rows' medians with
care.

**The hypothesis, answered by in-kernel cycle accounting.** Producer-side conversion does overlap
with MMA, **mostly**: running the full conversion on otherwise-idle warps with the dependency
removed costs the MMA warps 0–7 %, and a third converting warp left the MMA compute phase
unchanged. **The dominant loss was pipeline latency, not ALU:** with two TMA stages the MMA warps
waited 570–724 cycles per k-tile for converted tiles (16–21 %); a third stage cuts that to 72–83,
but fits the 101,376 B budget only with a shared-memory-free epilogue, which gives back 3–5 %,
netting +3–4 %. MMA-loop SASS is 1.97 non-MMA instructions per HMMA, against 1.78 for dense and
10.73 for v3: the conversion really has left the MMA warps.

**A correction to this doc.** It said mixed input "tops out at bf16-dense speed by construction".
The ceiling is bf16 tensor-core throughput, not a real dense kernel's speed: with the conversion
compiled out, this design ran **above** the dense sm120 kernel on three of four shapes, because an
NVFP4 k-tile streams 21.5 KB against 32 KB for bf16. A well-hidden conversion could therefore match
or edge past an actual bf16 kernel.

**Next step (agent):** software-pipeline the transform's item loop, whose four load→store chains
are serialised in the SASS, and trim the lookup sequence, for an estimated ~30 % cut in the
1700–2000-cycle convert phase; then a smaller-tile TMA epilogue that fits beside three stages.
Prototype limits unchanged: N % 128, K % 64, TN, batch 1, synthetic weights, not in MLX.

## Quiet-GPU rerun and stall attribution (2026-09-11 16:45–17:12)

GPU0 idle at 0 % with production unloaded; runs strictly sequential; per-step GPU state logged
(`claude-scratch/quiet-rerun.log`). All three harnesses now agree on cuBLAS within 0–13 %, so the
twofold gaps between earlier tables were contention and Triton's default *mean*, not harnesses.

**The sm120 kernels** (`preflight-runs/sm120-quiet-ab.log`, 5 rounds, spread 0–1 % on 7 of 8 rows,
40/40 validations with zero elements >5 % off):

| shape | M | tw2s3 | v3 | dense sm120 | cuBLAS | tw2s3/cuBLAS | tw2s3/dense | tw2s3/v3 |
|---|---|---|---|---|---|---|---|---|
| gemma4:31b gate | 2048 | 309 | 280 | 335 | 341 | 0.91 | 0.92 | 1.11 |
| gemma4:31b gate | 4096 | 310 | 279 | 338 | 348 | 0.89 | 0.92 | 1.11 |
| gemma4:31b down | 2048 | 303 | 266 | 309 | 336 | 0.90 | 0.98 | 1.14 |
| gemma4:31b down | 4096 | 280 | 247 | 302 | 327 | 0.86 | 0.93 | 1.14 |
| qwen3.8 gate | 2048 | 300 | 272 | 320 | 340 | 0.88 | 0.94 | 1.10 |
| qwen3.8 gate | 4096 | 299 | 268 | 326 | 340 | 0.88 | 0.92 | 1.11 |
| qwen3.8 down | 2048 | 288 | 254 | 298 | 322 | 0.89 | 0.96 | 1.13 |
| qwen3.8 down | 4096 | 280 | 246 | 300 | 312 | 0.90 | 0.93 | 1.14 |

The contended 0.71 row and the gemma "bimodality" were contention.

**MLX's own kernels** in the #286 harness (`preflight-runs/qqmm-quiet-final.jsonl`), TFLOP/s:

| shape | M | MLX `qmm` | dequant+bf16 | native FP4 `qqmm` | MLX bf16 | tw2s3 ÷ `qmm` |
|---|---|---|---|---|---|---|
| gemma4:31b gate | 2048 | 169 | 242 | 654 | 298 | 1.83 |
| gemma4:31b gate | 4096 | 176 | 298 | 876 | 342 | 1.76 |
| gemma4:31b down | 2048 | 152 | 239 | 579 | 301 | 1.99 |
| gemma4:31b down | 4096 | 151 | 292 | 651 | 334 | 1.85 |
| qwen3.8 gate | 2048 | 157 | 240 | 633 | 310 | 1.91 |
| qwen3.8 gate | 4096 | 163 | 291 | 783 | 334 | 1.83 |
| qwen3.8 down | 2048 | 156 | 228 | 564 | 304 | 1.85 |
| qwen3.8 down | 4096 | 151 | 290 | 720 | 335 | 1.85 |

So tw2s3 is **~1.8× MLX's `qmm`** (1.6–2.0× by ratio to cuBLAS within each harness; the agent's
"~3×" was contention). MLX `qmm` is 0.45–0.57 of its bf16. At M=1 `qmm` stays best
(0.06–0.08 ms vs dequant 0.58–0.86, `qqmm` 0.12–0.15), so any new path belongs above a row
threshold.

**Triton prototype, ptxas 12.8, median timing:** 74–88 TF/s = 0.23–0.26× cuBLAS on all 8 rows —
the contended ratios held. The CUDA 13.0 half **did not run**: Triton 3.3.1's
`ptx_get_version()` has no branch for CUDA 13 and raises. The script now accepts
`TRITON_FORCE_PTX_VERSION` (passes `ptx_version` to the kernel, skipping the check) and the rerun
pins PTX 87 for *both* halves, so they differ only in the assembler. Inference meanwhile: the
sm120 kernels were built with the same 12.8.61 compiler and reach 0.9× cuBLAS, so 12.8 is not
what limits the Triton prototype.

**Stall attribution** (Nsight Compute 2025.3.1 under sudo, `--clock-control none`, one launch
each, gemma4:31b gate M=2048; reports `preflight-runs/ncu-*-gemmagate2048.ncu-rep`):

| metric | tw2s3 | v3 | dense |
|---|---|---|---|
| duration | 1.51 ms | 1.64 ms | 1.37 ms |
| tensor pipe active, % of peak | 80.5 | 74.9 | 91.8 |
| issue slots used, % | 29.0 | 56.2 | 17.4 |
| instructions executed | 664 M | 1,387 M | 349 M |
| LSU pipe, % of peak | 19.4 | 42.4 | 17.6 |
| stall: barrier (warps per issued inst.) | 0.48 | 0.23 | 0.18 |
| stall: lg_throttle | 0.34 | 0 | 0 |
| stall share: SYNCS / WARPSYNC | 16.9 % / 5.5 % | 13.9 % / 5.9 % | 12.9 % / 6.4 % |
| stall share: conversion ops (PRMT, LOP3, HMUL2, LDS, STS) | 7.3 % | 29.5 % | ~0.6 % |
| stall share: STG | 4.1 % | 0 | 0 |

Findings, and what they change:

- **Not issue-bound.** 29 % of issue slots are used, so trimming the conversion's lookup
  sequence (the previous agent's proposal, and the H100 worklog's main lesson) buys little.
- **The conversion is off the critical path.** Its stall share fell from ~30 % to ~7 %, and with
  three stages the transform warps already wait 788–1020 cycles per tile for ring space: faster
  conversion would not move wall-clock time much.
- **The measurable losses against dense are the output stage and synchronisation.** `STG` and
  `lg_throttle` exist only in tw2s3 — the register-to-global epilogue it needs to fit three
  stages — and barrier stalls are ~2.7× dense. Tensor-pipe activity is 80.5 % against 91.8 %.
- **`UIADD3` is a red herring.** It is 14.7 % of tw2s3's samples, but 21.2 % of dense's: the
  instructions are predicated-off padding (`@!UPT UIADD3 URZ, …`) whose "wait" stall is the HMMA
  dependency latency attributed to the next slot. Intrinsic to the mainloop; ignore it.

**Revised priorities for the next iteration:** (1) recover a TMA epilogue that fits beside three
stages — smaller epilogue tile, or K=32 tiles with four stages; (2) cut ring synchronisation —
fewer arrive/phase-check operations per tile; (3) only then the conversion loop.

## Acceptance criteria

1. ☑ **Bench after the campaign** (two runs, production still on the card): ratios repeatable
   at M ≥ 2048 (table above). ☐ The crossover row count and the documented default threshold
   (1024 is the placeholder; 2048 — full prefill chunks only — is the conservative choice if
   criterion 3's 1122-row shape shows no gain).
2. ✗ **FAILED: the flag changes output, deterministically.** The control settles it
   (`repeat-namebbox.sh`, `document_single` ×3 per flag state, one container each, rendered with
   `summarize_reps.py`):

   | model | flag off (n=3) | flag on (n=3) | within-state spread |
   |---|---|---|---|
   | gemma4:12b-nvfp4 | 0.714 | 0.622 | **0 — identical across all 3 runs, both states** |
   | qwen3.8:27b-nvfp4 | 0.542 | 0.697 | **0** |
   | qwen3.6:35b-a3b-nvfp4 | 0.504 | 0.613 | **0** |

   Each configuration is bit-reproducible on this arm; the two configurations differ far outside
   that spread. So the 7/35 T1 differences below are **all** attributable to the flag, not to
   noise — as a changed accumulation order should be: deterministic, and different. Two of the
   three models score higher with the flag on and one lower, so there is no evidence of a
   systematic quality change, but "output-preserving" is false and criterion 2 as written
   ("within run-to-run spread") cannot be met by any threshold. Anything built on this path has
   to be judged as a **numerics change**, with the vision suite as the gate.

   Prior detail, for the record. **Parity run.** Think-off T1 with the flag on (`main276dq_1_1_`, 5 models,
   0 OOMs, 0 errors) against the flag-off `main276_1_` cells: **7 of 35 quality cells and 1 of 40
   contract cells differ** (`preflight-runs/dequant-t1-render.md`). Four are trivial (scene IoU
   ≤ 0.004, one 7 px OCR hit). Three are large and all on `name_bbox`, the arm every earlier
   comparison found volatile — 12b 0.622 on / 0.714 off, qwen3.8 0.697 / 0.542, qwen3.6 0.613 /
   0.506 — in **both directions**, which is what a changed accumulation order looks like rather
   than a quality change. MLX think-off is usually reproducible cell for cell, but the `main`
   validation already caught one arm flipping between repeats of one build, so the spread is not
   zero and is unmeasured for this arm. ☐ The control that settles it: `document_single` ×3 with
   the flag off and ×3 with it on, one container each, on those three models
   (`repeat-namebbox.sh`).
3. ☑ **Prefill throughput** measured on all six shapes (table above): 1.16–2.13× on images,
   1.17–1.37× on long text.
4. ✗ **FAILED. Runner peak memory** rises up to **6.8 GiB** with the flag on (gemma4:31b, one
   image); qwen3.8 +0.8–1.2 GiB; long text unchanged. The dequantised copies are layers wide, not
   one layer. Bounding them is a prerequisite for any default.
5. ☐ With the flag unset: no behaviour change (the policy returns false at threshold 0).

## Layer 2, not in scope here: the FP4 GEMM (`qqmm`)

Same seam, a second threshold, activations quantised per call, the ModelOpt global scales wired
through, and an accuracy gate on the bbox and contract arms before it is offered even as an
opt-in. The bench prices it; the vision suite decides it. Turing excluded by construction.

## Side findings from the bench

- MLX caches its NVRTC output under `/tmp/mlx/<version>/ptx` unless `MLX_PTX_CACHE_DIR` is set;
  every fresh runner container recompiles — the 10–15 min cold first request. A standalone MLX
  tool also needs `CUDA_PATH`/`CUDA_HOME` and `LD_LIBRARY_PATH` pointing at the `mlx_cuda_v13`
  dir, which the server sets for its runner. Persisting the cache is a one-variable change for
  the campaign containers and a candidate server default — **one directory per GPU
  architecture**: the entries are named by kernel and shape only, and a directory shared
  between the 2080 Ti and the RTX PRO 6000 handed the Blackwell runner a Turing-compiled
  `qmm_naive` kernel; every gemma4 request aborted with `Failed to load compiled … kernel`
  (2026-09-08, the first attempt at criterion 3).
- The production llama-server runs its CPU pool at 16 threads (no `-t` passed) and busy-polls
  ~8 cores while serving a GPU-resident model; `num_thread` is a request/Modelfile option and a
  probe for it is staged (`probe-num-thread.sh`).
