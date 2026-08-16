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

Last clean build is b9990 (259ae1df8b52), first affected is b9992. b9991 is not tagged and the only other commit in that range is Vulkan-only.

Bisected by source, then checked at runtime on both sides of the boundary. Same GPU, same request, cold server, first request each time, `n_ubatch = 2048` and `n_tokens_batch = 2040` in every log:

| build | result |
|---|---|
| b9888 (ollama 0.32.1) | no fault, 3/3 cold runs |
| b10069 (ollama 0.32.2) | illegal memory access |
| b10353 | illegal memory access |

For anyone hitting this through ollama: v0.32.1 is the last release on a clean llama.cpp, v0.32.2 is the first affected.
