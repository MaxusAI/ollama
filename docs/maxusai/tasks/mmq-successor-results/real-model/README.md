# Real-model reproduction of the MMQ MUL_MAT_ID over-read (2026-10-05)

qwen3.6:35b-a3b-q4_K_M, one 3072x1728 image (vision-suite ladder image), num_predict=1, num_batch=2048 so the
image prefills as 2048+2032-token ubatches through the MoE experts. CUDA host, sm_120, b11081 payload, GPU0.
Each row is one cold container; only libggml-cuda.so differs between rows.

| payload (libggml-cuda.so) | src1 allocation | num_ctx | outcome |
|---|---|---|---|
| raw (upstream `ne11`, 903 removed)        | VMM pool (shipping)   | 8192  | HTTP 200, image decoded, 1 token |
| raw (upstream `ne11`, 903 removed)        | VMM pool (shipping)   | 33792 | HTTP 200, image decoded, 1 token |
| raw (upstream `ne11`, 903 removed) + 910  | exact-size cudaMalloc | 8192  | **illegal memory access at `decoding image batch 1/2`, runner core-dumped (HTTP 500)** |
| fixed (new 903 widest-tile) + 910         | exact-size cudaMalloc | 8192  | HTTP 200, both batches decoded cleanly, 1 token |

- **The raw read is real and in the image-prefill MoE matmul.** With `ne11` padding, the broadcast gate/up
  MUL_MAT_ID (ne11==1) gets zero tail padding; the kernel reads a full tile past `src1_q8_1`. Crash backtrace:
  `libllama.so: llama_context::process_ubatch -> llama_context::decode`, at `ggml-cuda.cu:108` (CUDA_CHECK) during
  the image-batch decode.
- **The shipping VMM pool masks it.** On the stock pool the over-read lands in memory the pool has already mapped,
  so the run completes (HTTP 200) at both num_ctx 8192 and 33792. The over-read still happens; it is latent. This
  is why the exact-size allocation (debug patch 910, MMQ_EXACT=1) is needed to expose it deterministically, and why
  it was originally seen only as an intermittent crash whose trigger looked like "num_ctx > 32768".
- **The widest-tile 903 fixes it.** Same exact-size allocation, same image: the allocation now includes the tile,
  the read stays in bounds, and the decode is clean.
- **compute-sanitizer through ollama:** ollama re-execs its runner with a rebuilt environment that drops
  compute-sanitizer's injection, so wrapping `ollama serve` instruments only the parent. The precise
  "Invalid __global__ read ... mul_mat_q" line is from the controlled test-backend-ops path
  (`../stock-master-dd266785c.tsv`, exact runs). On the real model the exact-size allocation turns the same read
  into a deterministic illegal-access crash, shown above.

910-mmq-exact-debug.patch is a DEBUG patch (env-gated MMQ_EXACT), not part of 903 and not shipped. It exists only to
give src1 its own allocation so the read is caught without the VMM pool masking it.
