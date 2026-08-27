# Upstream PR draft: llm: force fp32 cuBLAS accumulation for qwen2.5-vl runners

**Prepared 2026-08-27; claims tightened as measurements landed.** Branch `qwen25vl-cublas-f32-accum` (this fork, commit 833e4a1f) carries the 90-line adaptation onto upstream `ollama/ollama` main (13f2fb8c): `go test ./llm/` green, gofumpt clean.

**Attach with the PR:** `docs/maxusai/vision-suite/synthetic-triggers/trigger_checker56_1350x1800.png` (12 KB, md5 `afc8ff7e84ee8958878b44675565d5b0`) — verified as the committed file on stock 0.30.0 and 0.33.0.

**Companion llama.cpp patch (optional, one call):** `llama/compat/905-fix-clip-ffn-down-prec-f32.patch` on branch `fix/clip-mm-prec-f32` — sets `GGML_PREC_F32` on the FFN down-projection, the node the tracer named. Validated end to end.

Measured-claims inventory: checkerboard verified on stock 0.30.0 + 0.32.9 + 0.33.0; corpus triggers on 0.30.0/0.32.9/0.32.15/0.33.0; 0.24.0 falsified as clean by a reproducible synthetic; 0.7.1 affected (CPU healthy / GPU garbage) but resists 199 synthetics due to its ~1 MP token cap; failing node localised to `ffn_down-31` (3 inf of 15.7M).

File with:

```bash
gh pr create --repo ollama/ollama --head MaxusAI:qwen25vl-cublas-f32-accum --base main --title "llm: force fp32 cuBLAS accumulation for qwen2.5-vl runners" --body-file docs/maxusai/upstream-qwen25vl-f32-accum-pr.md
```

Everything below the rule is the PR body, verbatim.

---

Fixes deterministic `'?'×31` garbage decode from qwen2.5-vl on CUDA by forcing fp32 cuBLAS accumulation in that family's runner. 90 lines, Go-only, no behavior change for any other model. Related: #14170, #17687.

## The bug, in one minute on stock ollama

```python
import numpy as np
from PIL import Image
ys, xs = np.mgrid[0:1800, 0:1350]
cells = ((xs // 56) + (ys // 56)) % 2
Image.fromarray(np.where(cells[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)).save("trigger.png")
```

(An identical prebuilt PNG is attached to this PR — 12 KB, md5 `afc8ff7e84ee8958878b44675565d5b0`, 1350×1800 — so no dependency on numpy/Pillow versions. Being exactly specified integer pixels it has no decoder ambiguity: the generator and the file reproduce bit-identically.)

Send `trigger.png` to stock `qwen2.5vl:3b-q4_K_M` (`/api/chat` or `/api/generate`, `temperature 0`), fully GPU-resident (measured on stock 0.32.9; the same clip vision path with the same fp16-accumulate GEMMs serves every release since 0.30):

| serving | result |
|---|---|
| default | `'?'×31`, `done_reason: null` — every call, every quantization (q4_K_M / q8_0 / fp16) — verified with this exact file on stock **0.30.0** and **0.33.0** |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (or `bf16`) | correct description of the checkerboard |
| CPU-only | correct |

Ordinary photos trigger it too — a corpus of insurance photos surfaced the class, and those corpus triggers reproduce on stock **0.30.0, 0.32.9, 0.32.15 and 0.33.0**. Large high-contrast images are the worst case, so 1800–2048 px OCR-style pipelines are maximally exposed. On 0.32.10–0.32.15 the first poisoned request additionally leaves the runner slot returning garbage for **every subsequent request** until reload (other releases recover on the next request).

Releases before 0.30 served this family through the since-removed Go vision engine. That implementation is **not** clean either — it carries the same fp16-accumulate defect with its own *disjoint* trigger set, because fp16 overflow depends on GEMM summation order, so each implementation picks different victims. Measured on a 768-image production fold: 0.7.1 fails at row 8 (a different corpus image garbles it on request #1 and poisons its slot) and 0.24.0 — which passes both spot-probe images — fails at row ~172 on a member of its own set. Passing a spot check is not passing a corpus; no engine is clean.

The 0.30 boundary is about *fixability*, not the defect: `GGML_CUDA_CUBLAS_COMPUTE_TYPE` does not exist in the pre-0.30 engines' ggml (binary grep: 0 matches in 0.7.1/0.24.0 with `GGML_CUDA_FORCE_MMQ` as a positive control; 2 matches in current builds), so llama-server-era releases are runtime-fixable and the removed engines never were. This checkerboard targets the clip path those releases share; the fix direction — ≥fp32 accumulation for this tower's matmuls — applies to any fp16-accumulate implementation of it.

## Mechanism (measured, not inferred)

Hooking every stage of the HF `Qwen2.5-VL-3B-Instruct` vision tower in bf16 shows the final block (`blocks.31.mlp`) emits massive-activation outliers at **0.57× fp16's 65,504 ceiling for ordinary images** — every input walks the cliff edge. Trigger images push those channels to 0.77–1.07×. The stored tensors still fit in fp16; the **partial sums inside the next fp16-accumulate GEMM** transiently exceed the ceiling before cancellation → inf → NaN cascade → degenerate decode. The generated checkerboard above measures 1.06×.

On the CUDA backend, the vision tower/merger's f16-weight matmuls run through `ggml_cuda_mul_mat_cublas` with `compute_type = src0->type`, i.e. `CUBLAS_COMPUTE_16F`; the small-batch `mmf`/`mmvf` kernels already accumulate fp32, so cuBLAS is the only fp16-accumulate stage — which is why the failure is CUDA-specific, quant-independent (mmproj weights are f16 in every quant), and immune to `f32`/`bf16` compute and to CPU. Flash-attention on/off, KV cache f32/bf16, and `GGML_CUDA_FORCE_MMQ` were all swept and change nothing: the overflow is in the weight GEMMs those paths share. The margin (~40% between ordinary and triggering images, on a tower idling at 0.57×) is structurally too thin for fp16 accumulation to be safe for this family.

## The change

ggml already exposes `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, and each llama-server runner serves exactly one model — so setting `=f32` in the qwen25vl runner's subprocess environment scopes fp32 accumulation to the affected family:

- `applyArchServerEnvs()` in `llm/llama_server.go` (+36, beside the existing arch-keyed `qwenVLServerArgs`), called from the runner-env assembly.
- An operator-set `GGML_CUDA_CUBLAS_COMPUTE_TYPE` always wins: `=f16` restores stock behavior for A/B; `=bf16` also heals (cuBLAS bf16 GEMMs accumulate in fp32) at timing parity with f32, measured on both Turing (sm_75, no native bf16 tensor cores — cuBLAS still takes the path) and Blackwell.
- Table test (+54) covering the default, the override, and untouched arches.

Cost: vision-encode GEMMs for this family run f32 (weights dequantized + `cublasSgemm`); warm trigger-image encode measured 676 ms (Blackwell) / 1059 ms (Turing) under f32 versus a garbage result under f16. Text-side cost is ~nil for quantized tags (MMQ/MMVQ carry those matmuls; the f16 decode-vector kernels already accumulate fp32); f16 text weights pay an fp32-GEMM prefill cost — scoped to the qwen25vl family only.

`qwen2vl` shares the graph builder but has no measured trigger and is deliberately left stock; the switch extends trivially if one shows up.

## Validation

- Causal three-way A/B on stock 0.33.0: default garbles, `f32`/`bf16` heal, forcing `f16` reproduces — same matrix on Turing and Blackwell with this patch's gate active (default heals; `=f16` override brings the garbage back; `=bf16` override heals).
- Slot follow-ups stay clean after a healed trigger request.
- `go test ./llm/` green; `gofumpt` clean.

## Prior art

The model-level fragility is documented: [huggingface/transformers#33294](https://github.com/huggingface/transformers/issues/33294) reports Qwen2-VL in fp16 producing *"gibberish output (repeated exclamation marks)"* alongside `probability tensor contains either inf, nan or element < 0`, with fp32 clean — the same fingerprint we see. It was fixed by PR #33312 (replace inf with zeros in attention weights), then corrected in [#35151](https://github.com/huggingface/transformers/issues/35151) because that fix also zeroed the `-inf` causal mask. Qwen's own guidance is that these models are trained in bf16 and fp16's range overflows, which matches our measurement that `bf16` heals as well as `f32`.

We found no report of this class in llama.cpp's clip/mtmd path. The closest is [ggml-org/llama.cpp#20081](https://github.com/ggml-org/llama.cpp/issues/20081) — Qwen3.5-27B mmproj giving drastically wrong vision output on Vulkan versus CUDA, reliably and only for *specific images*, with no precision cause identified and still unresolved. That is the same phenomenon shape on a different backend and may well be the same root cause.

## Scope notes for review

- **Why not fix clip.cpp?** It should also be fixed there, and we have localised it precisely. Instrumenting the clip graph with a non-finite eval callback names the failing node on pin `b10488`:

  ```
  node #1106  name='ffn_down-31'  op=MUL_MAT  type=f32
  shape=[1280,12288]   bad=3/15728640   nan=0  inf=3
  src[0]: v.blk.31.ffn_down.weight  f16 [3420,1280]
  src[1]: ffn_swiglu-31             f32 [3420,12288]
  ```

  **Three elements out of 15.7 million** overflow to `inf` — enough to NaN the image embeddings (`MTMD_DEBUG_EMBEDDINGS` reports `mean=nan, sum=nan` on GPU and clean stats on CPU) and produce the degenerate decode. So the minimal llama.cpp fix is a single call in `clip_graph::build_ffn`:

  ```c
  if (down) {
      cur = build_mm(down, cur);
      if (cur->op == GGML_OP_MUL_MAT) {
          ggml_mul_mat_set_prec(cur, GGML_PREC_F32);   // FFN down-proj consumes the activation peak
      }
  }
  ```

  Validated end to end with no global precision forcing on either side: stock `libmtmd` garbles the checkerboard, this one call yields a correct description and zero non-finite nodes. We are happy to file that as a separate llama.cpp PR. This ollama-side PR is still worth taking on its own: it protects users on every currently released llama.cpp pin, needs no native-code change, and reverts in one line once an upstream fix lands.

  *A/B caveat for reviewers:* `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` **overrides** per-op `GGML_PREC_F32` (the env check runs after the `op_params` check in `ggml_cuda_mul_mat_cublas`), so testing the per-op fix with `=f16` produces a false negative. Use `=auto`.
- **Why f32 and not bf16 as the default?** Both heal; f32 is closest to CPU numerics and measured at parity. bf16 remains one env var away for any deployment that prefers it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
