# qwen35moe MUL_MAT_ID crash: the full investigation

2026-08-13, CUDA host 10.8.0.6 (RTX PRO 6000 Blackwell, sm_120). From first symptom to
one-line fix, including every hypothesis that died and why. The dead ends are the point of
this document — three of them were built and tested before the real cause was found, and each
died of the same methodological error.

**Outcome:** `ggml/src/ggml-cuda/mmq.cu` sizes the MMQ tail padding from `ne11` instead of the
flattened row count `ne12 * n_expert_used`. In the MoE broadcast case `ne11 == 1`,
`ggml_cuda_mmq_get_J_max()` returns 0, so the quantised buffer gets no tail padding while the
kernel overruns it by up to a 512-row tile. Fixed in `llama/compat/903-fix-mmq-ids-padding.patch`.

## Symptom

```
ggml_cuda_compute_forward: MUL_MAT_ID failed
CUDA error: an illegal memory access was encountered   (ggml-cuda.cu:2374)
```

`qwen3.6:35b-a3b-q4_K_M` + an image + `num_ctx > 32768` → HTTP 500, runner core-dumps,
ollama restarts it. Present on every payload from `cb295bf59` (0.32.0) through `f8def7fe1`,
and on stock `ollama/ollama` images. Not a fork regression.

## Hypotheses, in order

### H1 — a regression in our fork or a newer llama.cpp payload

**Died:** stock `ollama/ollama` images crash identically, as do three different llama.cpp
payloads. An early "0.32.1 is clean" result was a **false pass** — the probe changed `num_ctx`
but the model never reloaded (20 s reply against a 195 s cold load), so it measured the
previous window. *Lesson: a pass that returns suspiciously fast is not a pass.*

### H2 — VRAM, alignment, generation length, dense architectures

**Died:** 85 GB free at crash; `32769` (1 mod 256) fails like `33792` (0 mod 256);
`num_predict=1` reproduces; gemma4 is clean at the same context with an image.

### H3 — cap `num_batch` below the image token count

Not a hypothesis so much as a workaround, and it worked. Rejected as a fix on the grounds
that it must be recomputed per image — which was correct, and which is what pushed the
investigation toward instrumentation.

### H4 — `mmvq.cu:517`, the `channel_y` broadcast clamp

The theory: `channel_y = ncols_dst == 1 && ids ? fastmodulo(...) : channel_dst` wrongly gates
the `nchannels_y` clamp on `ncols_dst == 1`, letting `channel_y` over-read a single-channel
`y`. It had a corroborating sibling — `mul_mat_vec_q_moe` applies the modulo unconditionally.

**Built and tested. Crash unchanged.** The patch was genuinely compiled in (`.so` hash moved
`e5fa9a96…` → `0dfc81c8…`) and genuinely loaded (`libdirs=ollama,cuda_v13`, and the crash
message carries the `/build/llama-server-cuda_v13/` path).

**Dead twice over:**

1. `mmvq.cu:905-913` intercepts exactly `has_ids && ncols_dst > 1` and routes it to a
   different kernel, so the patched branch is unreachable. The `ncols_dst == 1` guard is
   correct by construction.
2. mmvq entry is gated on `ne2 <= MMVQ_MAX_BATCH_SIZE` (8), and the crash is on a 2040-token
   ubatch. That file is not involved at all.

*Lesson: the hypothesis explained a shape signature and none of the other known facts — not
the apparent 32768 boundary, not the `num_batch` workaround, not persistence across a year of
llama.cpp revisions. It should not have been built.*

### H5 — CUDA graphs

**Died by experiment:** `GGML_CUDA_DISABLE_GRAPHS=1` still crashes, with the env var verified
present in the runner process (`cmd.Env = os.Environ()`, `llm/llama_server.go:446`). Useful
anyway — it made per-node attribution trustworthy for the instrumented run.

### H6 — the hybrid gated-DeltaNet/SSM block

Ranked first by a post-mortem, on the reasoning that `qwen35moe` is a hybrid architecture with
its own `GGML_OP_GATED_DELTA_NET` op and a recurrent memory, and that this was the only
candidate surviving fusion-off and `FORCE_CUBLAS`.

**Died:** the instrumented fault named `MUL_MAT_ID` at layer 3's MoE gate. Never the
delta-net.

### H7 — MMQ ids-path padding — **correct**

Found by instrumenting rather than reasoning. See below.

## The two facts that were wrong the whole time

**The "32768 threshold" was never a llama.cpp property.** `server/sched.go:880-889`
(`generationBatchForContext`) sets the batch to 2048 above 32768, 1024 above 4096, else 512,
and `llm/llama_server.go:590-591` passes that value as **both** `-b` and `-ub`. Raising
`num_ctx` past 32768 silently raises the ubatch to 2048, so `num_ctx` and ubatch were
perfectly confounded in every experiment until the cell was sampled directly:
**`num_ctx=8192` with `num_batch=2048` crashes.** This is upstream ollama (PR #16031,
>= v0.30.0), not our fork.

Consequence: upstream llama.cpp's default `n_ubatch` of 512 splits a 2040-token image into
four chunks, so upstream rarely produces the shape at all. That is why this looked like an
ollama bug and why upstream reports of it were sporadic.

**`MUL_MAT_ID failed` was a detection point, not a location.** `ggml-cuda.cu:2371` is a bare
`cudaGetLastError()` after dispatch — asynchronous and sticky, with CUDA graphs on top. For
most of the investigation nobody knew which kernel actually faulted.

## What finally worked

A build with three changes (`llama/compat/902-debug-fault-locate.patch`):

1. `cudaStreamSynchronize(ctx.stream())` immediately before the existing `cudaGetLastError()`,
   so the error is attributed to the node that caused it.
2. A `MAXUSAI_FAULT` line printing the faulting op, tensor name, and the dims of `dst` and
   every `src`.
3. `MAXUSAI_MMID` markers at **all five** `mul_mat_id` dispatch exits, each printing the `ids`
   pointer and the real dims.

Point 3 fixed both defects of the earlier instrumentation, which logged one branch only and
never printed the pointer — so "ids != nullptr on the crashing call" had been an inference
dressed up as a measurement.

Run cold, target request first, with a **passing** case as a baseline for comparison:

| case | branch | ne2 | ne11 | result |
|---|---|---|---|---|
| 1032-token image | `mmq` | 1032 | 1 | 80 calls, clean |
| 2040-token image | `mmq` | 2040 | 1 | **fault** |

```
MAXUSAI_FAULT op=MUL_MAT_ID name=ffn_moe_gate-3
  dst_ne=[512, 8, 2040, 1]
  src0=blk.3.ffn_gate_exps.weight : q4_K [2048, 512, 256, 1]
  src1=attn_post_norm-3 (reshaped): f32  [2048,   1, 2040, 1]
  src2=ffn_moe_topk-3             : i32  [   8, 2040,    1, 1]
```

The baseline case was suggested by the user and turned a single fault line into a controlled
comparison: same branch, same types, one dimension different.

## The bug

`mmq.cu:205-206`, the `ids` branch of `ggml_cuda_mul_mat_q`:

```c
const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
    ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
```

The data term uses `ne12 * n_expert_used` — named `ne11_flat` twelve lines below. The padding
term uses `ne11`. And `get_J_max` (`mmq.cuh:360`) with `ne11 == 1` computes
`ret = min(1, 512) = 1`, then `ret -= 1 % 8` → `0`, skips the loop, and **returns 0**.

`dedup_bcast = ne11 == 1 && n_expert_used > 1` (`mmq.cu:193`) is exactly the MoE gate/up case:
one token's activations feed 8 experts. The `!ids` branch is fine, because there `ne11` really
is the row count.

**Fix:**

```c
ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12*n_expert_used)
```

## Why it presented as intermittent

The overrun is a **fixed size** past the end, so it faults only when it crosses an unmapped
page — which depends on the pool's allocation history. Measured in one container:

- cold, 2040-token image, first request: **crash**
- the identical request after the pool had served a 4080-token image: **passes**

This also invalidated a whole bisection: image sizes 1032 → 1947 were tested ascending in one
warm process and all "passed", but every one of those passes is uninterpretable. Only cold,
first-request measurements mean anything. The user caught this.

It is also why `test-backend-ops` is not an oracle here: the over-read lands in padding rows
whose results the kernel discards, so NMSE is unaffected. **Output is byte-identical with and
without the fix** (verified on a non-crashing size), so the symptom is the crash alone — there
is no silent corruption, and prior benchmark numbers stand.

## Verification

Cold container, target request first, four consecutive runs, no fault — with the
instrumentation retained to confirm the **same `mmq` branch with identical shapes** still
executes, i.e. the fix corrects the path rather than avoiding it. Repeated across independent
containers and against an unfixed control built from the same tree.

## Methodological lessons

1. **Every load-bearing number appears verbatim in a log line, or it is marked INFERRED.**
   Three consecutive hypotheses died from reasoning about a value instead of printing it.
2. **A hypothesis must explain all known facts before it earns a build.** H4 explained one.
3. **A pass is not evidence** when the failure mode is state-dependent. Cold, first-request,
   fresh container — and always run an unfixed control alongside.
4. **Verify the instrument before trusting the measurement.** Check the env var reached the
   process, the payload actually loaded (`libdirs=`), and the patch is really in the binary —
   `llama-server --version` cannot tell you (see [[fast-cuda-dev-loop]]).
5. **A fast loop changes which methods are rational.** At 90 minutes per build the incentive
   is to reason hard and build once, which is exactly how an hour was spent confirming a wrong
   idea. At 8 minutes, instrumenting first is cheaper than thinking harder.

## Artefacts

- `llama/compat/903-fix-mmq-ids-padding.patch` — the fix
- `llama/compat/902-debug-fault-locate.patch` — the instrumentation (do not ship: it
  synchronises on every node)
- `docs/maxusai/tasks/rocm-mmq-ids-padding-test.md` — brief for testing HIP, which compiles
  the same `mmq.cu`
- `docs/maxusai/tasks/qwen35moe-mmq-testcases.cpp` — `test-backend-ops` cases at the live
  shapes, to be run under `compute-sanitizer`
