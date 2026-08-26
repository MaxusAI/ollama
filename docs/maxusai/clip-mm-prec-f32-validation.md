# clip mm prec f32 — validation runbook for patch 904

**Status: PREPARED 2026-08-26, awaiting the PR #214 release-level A/B before
runtime validation.** Compile validation done; no GPU probe has been run for
this patch yet.

Fix under test: `llama/compat/904-fix-clip-mm-prec-f32.patch` — force
`GGML_PREC_F32` on every f16-weight `mul_mat` in the mtmd/clip graph, so
CUDA/HIP accumulate those GEMMs in fp32 like the CPU backend does. Targets the
PR #214 finding: *the one fp16 stage common to every failing CUDA config and
absent from the healthy CPU path is the f16-weight matmuls of the vision
tower/merger (fp16-accumulate GEMM on CUDA, fp32 on CPU)*. Diagnosis:
`docs/maxusai/qwen25vl-3b-poison-image-garbage-decode.md`.

Mechanism, verified against the pinned `b10488` source:

- `ggml_cuda_mul_mat_cublas` sets `compute_type = src0->type`, so f16 weights
  run `cublasGemmEx` with 16F compute — fp16 partial sums. With
  `GGML_PREC_F32` it converts src0 to f32 and runs `cublasSgemm` — fp32
  inputs and accumulation, matching CPU numerics.
- The vision tower's GEMMs (`ne11` = thousands of patch positions) always land
  in the cuBLAS path: `mmf`/`mmvf` only take `ne11 ≤ 16`, and both already
  accumulate in fp32 (`float sum[...]` in `mmf.cuh`), so there is no
  small-batch gap.
- The flash-attn node already carries `GGML_PREC_F32` upstream; the non-FA
  `kq`/`kqv` matmuls have f32 activations as src0 and were never in 16F
  compute. That is why the FA/KV knob matrix in PR #214 could not heal the
  garbage — the overflow is in the weight matmuls this patch covers.

## Probe shape (shared by all steps)

3-request H/X pattern against a fresh container, identical to the PR #214
repro: known-good image → poison image (`02c9d7e1…`, 756×1008) → known-good
image, `/api/chat` with `num_ctx` 8192, `temperature` 0, `num_predict` 250,
model `qwen2.5vl:3b-q4_K_M`, read-only store bind. H = sensible labels,
X = `'?'×31` with `done_reason: null`. Expected patterns: broken = `HXH` or
`HXX`; fixed = `HHH`.

**Scheduling discipline:** do not run while another probe or campaign holds
the GPUs (the release A/B container `qwen-release-probe0` was resident when
this was written). Tear down probe containers in the same script that starts
them, and verify `nvidia-smi` is clean afterwards.

## Step 0 — zero-rebuild hypothesis test (run this first)

The pinned `b10488` ggml already ships a global override:
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` forces F32 compute for **all** cuBLAS
matmuls (`ggml_cuda_mul_mat_cublas`, ggml-cuda.cu). The PR #214 knob matrix
("workaround status: none reachable via environment") predates finding this
knob — it was never swept. It reaches llama-server through plain env
pass-through, so the hypothesis is testable on the **current failing image**
with no build at all:

```
docker run -d --rm --name clip-prec-probe --gpus '"device=<n>"' \
  -e OLLAMA_MAX_LOADED_MODELS=1 \
  -e GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 \
  -v <store>:/root/.ollama:ro -p 127.0.0.1:<port>:11434 <current-image>
```

Decision table:

| Result | Reading |
|---|---|
| `HHH` | Hypothesis **confirmed**: fp16-accumulate cuBLAS GEMMs are the overflow site. Ship 904 (same effect, scoped to clip graphs only instead of every text-model GEMM). |
| `HXH`/`HXX` | Hypothesis **falsified** for the cuBLAS-compute framing — do not ship 904 as the fix; next suspects are outside `mul_mat` compute type (conv/im2col materialization, soft-max, projector translation). |

Also worth one cell: `GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` — if healthy too,
that confirms the range-not-mantissa theory and documents a cheaper compute
mode upstream could prefer.

Note the knob is process-wide (it also moves f16 *text* GEMMs to f32), so it
is a probe and an emergency mitigation, not the fix; 904 scopes the change to
the clip graph.

## Step 1 — compile validation (done 2026-08-26)

`cmake -S llama/server --preset cpu` fetches b10488, applies 001→904 in order,
and `cmake --build build/llama-server-cpu --target llama-server` links clean
(Ubuntu 24.04, gcc 13). The patch is a post-pass at the two graph-build sites
(`reserve_compute_meta`, `clip_encode`) plus one helper — no insertion-point
overlap with patches 001/002/004/005.

## Step 2 — runtime A/B without a CUDA image rebuild

`clip.cpp` compiles into `libmtmd.so` (CPU stage artifact); `GGML_PREC_F32`
is graph-side `op_params` that the **existing** CUDA backend `.so` already
honors. So a patched `libmtmd.so` dropped into the current image is a full
runtime test — no nvcc, minutes not hours:

1. Build only the CPU stage: `docker buildx build --target llama-server-cpu …`
   (build the lib inside the image's own base so glibc matches — do not bind a
   host-built Ubuntu `.so` into the AlmaLinux runtime image).
2. Run the current CUDA image with the patched lib bound over the image's
   `/usr/lib/ollama/libmtmd.so.0` (`:ro`; SONAME is `libmtmd.so.0`, the real
   file is `libmtmd.so.0.<ver>` — match whichever the image ships).
3. Confirm the patch is live in the runner log at model load:
   `clip_graph_force_mm_prec_f32: f16-weight matmul accumulation: forced fp32`.
4. Probe: expect `HHH` on the poison sequence.
5. Causality check: same container with `-e OLLAMA_CLIP_MM_PREC=f16` → log
   says `f16 (legacy, …)` and the probe must regress to `HXH`/`HXX`.
6. Cross-check on both GPUs (Blackwell + Turing) and once on CPU-only
   (`NVIDIA_VISIBLE_DEVICES=void`) to confirm CPU stays H.

## Step 3 — full rebuild, preflight, and baselines

- Full image build as usual (CUDA stages unchanged by 904 but rebuild anyway
  for a deployable tag; buildx state on the 8TB array).
- Run the pre-deploy regression harness (`docs/maxusai/vision-suite/preflight/`).
  **Expect vision-check drift on CUDA hosts:** 904 changes numerics for every
  mtmd model (they now match the CPU path). Treat changed vision expected
  values as NEEDS_BASELINE re-recording, not as regressions — but *scores
  should move within noise*; a grading-level drop is a real finding.
- Perf: measure image-encode time before/after through the standard harness
  (`run_engine_compare.sh` arm via `TAG_PREFIX`/`REPEATS`, per the
  one-runner invariant). Worst case is a max-budget dynres cell (nemotron
  3328-token images). Expected cost: the f16 GEMMs become f32
  (`cublasSgemm` + one weight-dequant per matmul call) — encode-only;
  decode throughput must be unchanged. Transient VRAM bump is pool-managed
  (~tens of MB peak, largest converted weight is the 5120×2048 merger).

## Scope — what 904 does NOT address

- The **separable slot-poisoning** (FA-linked sticky garbage on Blackwell
  until reload). If fp16 overflow is the only garbage source, poisoning has
  nothing to stick to — but the runner-level recycle-on-degenerate-decode
  guard from the report is still worth having.
- The **Turing + `--flash-attn off` crash-loop** (`unexpected EOF` on any
  image request) — independent defect.
- The **Go-engine (0.7.1-era) disjoint trigger set** (`04431b0d…`): current
  builds serve GGML models via llama-server only, so 904 covers the serving
  path in use; the class in the deleted Go engine is historical. The MLX
  runner is a separate numerics domain, untouched.
- **Upstream**: 904 is upstream-reportable evidence (relates to ollama#14170,
  ollama#17687, and clip.cpp's commented-out KQ F32 override). The upstream
  form would likely drop the `OLLAMA_CLIP_MM_PREC` escape hatch or rename it
  `MTMD_*`; per the 9xx-band rule the patch leaves when the pin moves past an
  upstream fix.

## Relation to the pending release A/B

Whatever `0.24.0` vs `0.30.0` shows, it now reads as a per-image
trigger-set boundary (see the doc's correction section), not the introduction
point of the class — 904 targets the clip path every current build serves
with, so the prepared fix is valid under any A/B outcome. The A/B result
chooses the *narrative* for the upstream filing, not the fix.
