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

## Acceptance criteria

1. ☑ **Bench after the campaign** (two runs, production still on the card): ratios repeatable
   at M ≥ 2048 (table above). ☐ The crossover row count and the documented default threshold
   (1024 is the placeholder; 2048 — full prefill chunks only — is the conservative choice if
   criterion 3's 1122-row shape shows no gain).
2. ☐ **Parity**: think-off T1 on the five nvfp4 models with `OLLAMA_MLX_PREFILL_DEQUANT_ROWS`
   set against the current `main276_1_` cells, through the ADR 0012 generators; every quality
   cell within run-to-run spread, contract matrices identical.
3. ☐ **Prefill throughput** on `main`, flag on vs off, same container: one image (1.2k tokens),
   three images (3.3k), the 41.6k-token text prompt on gemma4:31b and qwen3.8; report tok/s and
   the request time, not just the GEMM.
4. ☐ **Runner peak memory** (`peak memory` line) unchanged within 0.5 GiB with the flag on.
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
  the campaign containers and a candidate server default.
- The production llama-server runs its CPU pool at 16 threads (no `-t` passed) and busy-polls
  ~8 cores while serving a GPU-resident model; `num_thread` is a request/Modelfile option and a
  probe for it is staged (`probe-num-thread.sh`).
