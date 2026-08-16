This is a regression, and I was able to pin it to a commit.

It was introduced by 6eddde06a4f2 ("CUDA: refactor MMQ kernel configuration", #24127), build b9992. That commit changed the padding term in both allocation branches:

```diff
-            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
+            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
-        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
+        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
```

That is correct for the `!ids` branch, where `ne11` is the row count. In the `ids` branch the row count is `ne12*n_expert_used`, so passing `ne11` is wrong there.

The difference matters because the two functions have different failure modes. `get_mmq_x_max_host(cc)` only looks at the architecture and returns 128 or 64, so the old code always allocated some padding no matter what the shape was. `ggml_cuda_mmq_get_J_max()` looks at `ne11` as well, and for `ne11 == 1` it computes `min(1, 512) = 1`, then `1 - 1 % 8 = 0`, skips its loop and returns 0. So the broadcast case gets no padding at all rather than a bit too little.

This is not architecture specific. `ggml_cuda_mmq_get_J_max` never reads `cc` before the `ret -= ret % 8` step, so `ne11 == 1` reduces to zero padding everywhere. Compiled the real `mmq.cuh` and evaluated it on the host for `q4_K`, `fallback=false` — it needs no GPU, `get_J_max` and `get_config` are pure functions of `(type, J, fallback, cc)`:

| architecture | J for ne11=1 | highest ne11 still giving 0 |
|---|---|---|
| Pascal, Volta, Turing, Ampere, Ada, Hopper, Blackwell | 0 | 7 |
| Vega, RDNA2 | 0 | 7 |
| RDNA3, RDNA3.5, RDNA4, CDNA2, CDNA3 | 0 | 15 |

Zero on all of them. The AMD tensor-capable parts have a wider range because no valid config exists at `J = 8` there either, so the loop steps down to zero.

Whether a part actually reaches this code is a separate question, and I only read that one rather than running it — `ggml_cuda_should_use_mmq` queries `smpbo` so it needs a real device. For `q4_K` with `ne11 = 2040` and 256 experts it returns true on everything I traced except Volta, which needs `ne11 < 64` once fp16 MMA is available and so leaves the MMQ path entirely at this batch size.

Last clean build is b9990 (259ae1df8b52), first affected is b9992. b9991 is not tagged and the only other commit in that range is Vulkan-only.

Bisected by source, then checked at runtime on both sides of the boundary. Same GPU, same request, cold server, first request each time, `n_ubatch = 2048` and `n_tokens_batch = 2040` in every log:

| build | result |
|---|---|
| b9888 (ollama 0.32.1) | no fault, 3/3 cold runs |
| b10069 (ollama 0.32.2) | illegal memory access |
| b10353 | illegal memory access |

For anyone hitting this through ollama: v0.32.1 is the last release on a clean llama.cpp, v0.32.2 is the first affected.
