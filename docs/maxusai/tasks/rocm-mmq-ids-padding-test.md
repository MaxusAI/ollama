# ROCm/gfx1151: does the MMQ ids-padding under-allocation affect HIP?

We found and **fixed** an out-of-bounds read in llama.cpp's `ggml/src/ggml-cuda/mmq.cu`
that crashes MoE vision models on CUDA. The ROCm backend **compiles the same source file**,
so gfx1151 is very likely affected. This asks for that to be confirmed on the AMD host.

Confirmed on CUDA 2026-08-13 (RTX PRO 6000 Blackwell, sm_120, payload `f8def7fe1`), by
observing the fault directly rather than inferring it.

## The bug

`ggml_cuda_mul_mat_q`, the `ids` (MUL_MAT_ID) branch, `mmq.cu:205-206`:

```c
const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
    ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
```

The data term is sized for `ne12 * n_expert_used` rows — the flattened row count, named
`ne11_flat` twelve lines below. The **tail padding term is sized from `ne11`**.

For MoE gate/up projections the activations are broadcast across experts, so `ne11 == 1`.
And `ggml_cuda_mmq_get_J_max` (`mmq.cuh:360`) with `ne11 == 1`:

```c
int ret = std::min(ne11, int64_t(512));   // 1
ret -= ret % 8;                           // 1 - 1 = 0
for (; ret > 0; ret -= 8) { ... }         // body never runs
return ret;                               // 0
```

returns **zero**. The buffer gets no tail padding at all, while MMQ processes rows in tiles
of up to `J_max = 512` and overruns the logical end by up to one tile.

The non-broadcast case is under-allocated too, just less: `ffn_down` passes `ne11 = 8`, so
padding is sized for 8 rows instead of 16,320.

**Fix** (`903-fix-mmq-ids-padding.patch`, in this directory):

```c
ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12*n_expert_used)
```

## Observed fault, for comparison with whatever HIP does

Instrumented build (per-node `cudaStreamSynchronize` before `cudaGetLastError`, so the
attribution is real and not a sticky async error):

```
MAXUSAI_FAULT op=MUL_MAT_ID name=ffn_moe_gate-3
  dst_ne=[512, 8, 2040, 1]
  src0=blk.3.ffn_gate_exps.weight : q4_K [2048, 512, 256, 1]
  src1=attn_post_norm-3 (reshaped): f32  [2048,   1, 2040, 1]
  src2=ffn_moe_topk-3             : i32  [   8, 2040,    1, 1]
MAXUSAI_MMID branch=mmq ne0=512 ne1=8 ne2=2040 ne02=256 ne11=1 ne12=2040 type=q4_K
```

Model: `qwen3.6:35b-a3b-q4_K_M` (`qwen35moe`, 256 experts, 8 used), layer 3 MoE gate.

## Why it looks intermittent — read this before testing

The overrun is a **fixed-size read past the end**. It only faults when it crosses an
unmapped page, so whether it crashes depends on the CUDA/HIP pool's allocation history.
Concretely, measured on CUDA:

- cold container, 2040-token image: **crash**, reproducibly
- the *same request* after the pool had served a 4080-token image: **passes**

**A pass is therefore not evidence of absence.** Test cold, with the target request first.
Do not reuse a warm runner, and do not conclude "not affected" from a green run.

Note also that the results of the over-read land in padding rows the kernel discards, so
output is unchanged — verified byte-identical across the fix on a non-crashing size. The
symptom is the crash only, not silent corruption.

## Why ROCm is likely affected

`ggml/src/ggml-hip/CMakeLists.txt`:

```cmake
file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
```

That glob includes `mmq.cu`, so HIP compiles the identical sizing expression.

Unknown, and the most valuable thing to determine: whether AMD reaches this branch at all.
`ggml_cuda_should_use_mmq()` differs on AMD, and `ggml_cuda_mul_mat_id` has an AMD-specific
early return at `ggml-cuda.cu:1891`. If HIP routes these shapes elsewhere, the sizing bug is
present but unreachable.

## What to do

### 1. Record the environment

```bash
docker exec <ctr> sh -c '/usr/lib/ollama/llama-server --version'
rocminfo | grep -m1 gfx ; cat /opt/rocm/.info/version 2>/dev/null
```

Note the gfx1151 host is gated at the 0.32.1 base (`docs/maxusai/amd-upgrade-gate.md`). That
gate concerns the 0.32.5 payload's other problems and does not exclude this bug — the sizing
expression is present in every payload we have checked.

### 2. Determine the branch taken (most valuable output)

Apply `902-debug-fault-locate.patch` from `llama/compat/` on the CUDA host — it logs
`MAXUSAI_MMID branch=...` at all five `mul_mat_id` exits plus a `MAXUSAI_FAULT` line with
full tensor geometry. Rebuild and run a MoE vision request.

If HIP logs `branch=mmq` with `ne11=1` and a large `ne12`, it is affected whether or not it
crashes.

### 3. Reproduce

Needs a MoE vision model (`qwen3.6:35b-a3b` or equivalent) and an image large enough to fill
one ubatch. On CUDA, ollama sets `-b` and `-ub` to the same value, tiered by context
(`server/sched.go:880`), so `num_ctx > 32768` yields `-ub 2048` and a ~2040-token image
arrives unsplit. Either raise num_ctx or set `num_batch` explicitly:

```bash
curl -s http://<host>:<port>/api/generate -d '{
  "model":"qwen3.6:35b-a3b-q4_K_M","prompt":"Describe this image.",
  "images":["<base64 1920x1080 png>"],"stream":false,
  "options":{"num_predict":1,"num_ctx":40960,"temperature":0}}'
```

Confirm `n_tokens_batch = 2040` in the runner log before trusting the result.

### 4. Better than waiting for a crash: use a sanitizer

Because the fault is state-dependent, the reliable instrument is a memory checker, not a
crash:

```bash
compute-sanitizer --tool memcheck ./build/bin/test-backend-ops -o MUL_MAT_ID     # CUDA
```

The ROCm equivalent is `rocm-smi`-adjacent tooling or an ASAN-instrumented HIP build; if
neither is available, the instrumented branch log from step 2 plus a cold reproduction is
sufficient evidence.

`qwen35moe-mmq-testcases.cpp` in this directory holds `test-backend-ops` cases at the live
shapes. **Note they may PASS even when the bug is present**: the over-read lands in padding
rows whose results are discarded, so NMSE is unaffected. Use them under a sanitizer, not as
a pass/fail oracle.

## What to report back

- llama.cpp SHA, ROCm version, gfx target
- Which branch `MAXUSAI_MMID` shows for a MoE vision request, with `ne11`/`ne12`
- Whether a cold reproduction crashes, and at what ubatch
- Sanitizer output, if available
- Whether applying `903-fix-mmq-ids-padding.patch` changes any of the above
