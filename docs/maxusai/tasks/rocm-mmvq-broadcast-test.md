## Summary

We found a CUDA out-of-bounds read in llama.cpp's `mmvq.cu` that affects **MoE vision
models** (qwen3.6 / qwen35moe). The ROCm backend **compiles the same source file**, so
gfx1151 is likely affected too — but AMD takes a different dispatch branch, so it must be
tested rather than assumed.

This issue asks for that test on the gfx1151/ROCm host.

## The bug (established on CUDA, 10.8.0.6)

`ggml/src/ggml-cuda/mmvq.cu:517`:

```c
channel_y = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
```

The `nchannels_y` clamp is gated on `ncols_dst == 1`, but `y` is broadcast whenever
`nchannels_y < nchannels_dst`, independently of `ncols_dst`. For MUL_MAT_ID on a MoE with
broadcast activations, `channel_y` then runs `0..nchannels_dst-1` into a `y` holding ONE
channel, reading up to `(nchannels_dst-1) * stride_channel_y` bytes past the end of
`src1_q8_1`.

Live shapes captured from a running `qwen3.6:35b-a3b-q4_K_M` (instrumented build):

```
VULNERABLE (ffn_up/gate):  type=Q4_K  ne0=512  ne1=8  ne2={2,4,7}  ne10=2048  ne11=1   ne02=256
SAFE       (ffn_down):     type=Q6_K  ne0=2048 ne1=8  ne2={2,4,7}  ne10=512   ne11=8   ne02=256
```

Over-read is a fixed `7 * 2304 = 16,128` bytes against a `4.6-18 KB` allocation.

**Trigger:** `ids != nullptr && ncols_dst > 1 && nchannels_y < nchannels_dst`

Strong corroboration: the dedicated MoE kernel in the SAME file
(`mul_mat_vec_q_moe`, ~line 741) applies the modulo **unconditionally** and is correct.

**Crucially it is an over-READ, so it does not always fault.** Whether it crashes depends
on whether those 16 KB are mapped, which depends on CUDA pool history. "No crash" therefore
does NOT mean "not affected" — the activations may just be silently wrong.

## Why ROCm is likely affected

`ggml/src/ggml-hip/CMakeLists.txt`:

```cmake
file(GLOB GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
```

That glob includes `mmvq.cu`, so HIP compiles the identical faulty line.

## Why it might NOT be — please check this first

`ggml_cuda_mul_mat_id` has an AMD-specific branch (`ggml-cuda.cu:1891`):

```c
} else {
    if (GGML_CUDA_CC_IS_AMD(cc)) {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
        return;
    }
}
```

and `ggml_cuda_should_use_mmq()` behaves differently on AMD. So ROCm may route these shapes
away from `mul_mat_vec_q` entirely. **Determining which path AMD actually takes is the
single most valuable output of this task.**

## What to do

### 1. Record the environment

```bash
docker exec <ollama-container> sh -c '/usr/lib/ollama/llama-server --version'   # llama.cpp SHA
rocminfo | grep -m1 gfx ; cat /opt/rocm/.info/version 2>/dev/null
```

Note: the gfx1151 host is gated at the **0.32.1** base (`docs/maxusai/amd-upgrade-gate.md`).
That gate is about the 0.32.5 payload's *other* problems; it does not exclude this bug,
which we confirmed is present in every payload from `cb295bf59` (0.32.0/0.32.1) onward.

### 2. Reproduce the crash path

Requires a MoE vision model (qwen3.6:35b-a3b or equivalent qwen35moe). Send an image
request with `num_ctx > 32768` and default `num_batch`:

```bash
curl -s http://<host>:<port>/api/generate -d '{
  "model":"qwen3.6:35b-a3b-q4_K_M",
  "prompt":"<a long prompt>", "images":["<base64 1920x1080 png>"],
  "stream":false,
  "options":{"num_predict":1,"num_ctx":40960,"temperature":0}}'
```

On CUDA this reliably returns HTTP 500 with
`CUDA error: an illegal memory access was encountered`.

**Do a fresh container restart before each attempt** — outcome depends on pool history, and
a warm runner can pass where a cold one crashes. Also probe `num_ctx=32769` (crashes on
CUDA) and `32768` (clean on CUDA).

### 3. Determine which kernel AMD dispatches (the important bit)

Apply this instrumentation as `llama/compat/900-debug-mulmatid.patch` and rebuild — it logs
shapes and the chosen branch at `ggml_cuda_mul_mat_id` entry. The patch is attached in the
comments below / available from the CUDA host at
`/mnt/8TB_SN850X_RAID1_BTRFS/tmp/ollama-build` (it prints `MMID_DBG branch=...`).

If ROCm logs `branch=mul_mat_vec_q` with `ne11=1` and `ne2>1`, it is affected regardless of
whether it crashes.

### 4. Better: test correctness, not just crashes

`test-backend-ops` compares against a reference backend, so it detects the silent-corruption
case deterministically:

```bash
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1151 -DLLAMA_BUILD_TESTS=ON
cmake --build build --target test-backend-ops -j
./build/bin/test-backend-ops -o MUL_MAT_ID
```

with the 15 cases we derived from the live shapes (file available from the CUDA host:
`qwen35moe-testcases.cpp`). The decisive pair is `n=1` (expect PASS, guard applies) versus
`n=2` (expect FAIL). `b=false` should also pass.

## What to report back

- llama.cpp SHA, ROCm version, gfx target
- Which branch `MMID_DBG` shows for the vulnerable shapes
- Whether the crash reproduces, and at which `num_ctx` / `num_batch`
- `test-backend-ops -o MUL_MAT_ID` output, especially NMSE on the vulnerable rows
- Whether the sibling kernel `mul_mat_vec_q_moe` is used instead

## Status of the CUDA-side fix

Candidate one-line fix under test on the CUDA host:

```c
channel_y = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
```

matching what `mul_mat_vec_q_moe` already does. If it holds, we will propose it upstream to
ggml-org (the file is stock ggml — not ollama's compat layer), and this ROCm result decides
whether the report covers HIP as well.
