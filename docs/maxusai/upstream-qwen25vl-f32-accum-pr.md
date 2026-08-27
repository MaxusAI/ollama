# Upstream PR draft: llm: force fp32 cuBLAS accumulation for qwen2.5-vl runners

**Prepared 2026-08-27.** Branch `qwen25vl-cublas-f32-accum` (on this fork, commit 5a3806cd) carries the 90-line adaptation of PR #215's gate onto upstream `ollama/ollama` main (13f2fb8c): `go test ./llm/` green, gofumpt clean. File with:

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

Send `trigger.png` to stock `qwen2.5vl:3b-q4_K_M` (`/api/chat` or `/api/generate`, `temperature 0`), fully GPU-resident, on any stock release from 0.32.9 to 0.33.0:

| serving | result |
|---|---|
| default | `'?'×31`, `done_reason: null` — every call, every quantization (q4_K_M / q8_0 / fp16) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (or `bf16`) | correct description of the checkerboard |
| CPU-only | correct |

Ordinary photos trigger it too (a corpus of insurance photos surfaced it); large high-contrast images are the worst case, so 1800–2048 px OCR-style pipelines are maximally exposed. In some releases (0.32.10–0.32.15) the first poisoned request additionally leaves the runner slot returning garbage for **every subsequent request** until reload.

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

## Scope notes for review

- **Why not fix clip.cpp?** The durable fix arguably belongs in ggml-org/llama.cpp (e.g. `GGML_PREC_F32` on the clip graph's f16-weight matmuls — we have a working patch of that shape and can file it there). This PR protects ollama users today via a knob ggml already ships, with no native-code changes and a one-line revert path once an upstream llama.cpp fix lands.
- **Why f32 and not bf16 as the default?** Both heal; f32 is closest to CPU numerics and measured at parity. bf16 remains one env var away for any deployment that prefers it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
