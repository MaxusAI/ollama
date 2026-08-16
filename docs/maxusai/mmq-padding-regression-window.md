# The MMQ ids-padding defect: exact origin

Companion to [[qwen35moe-mmq-investigation]]. Establishes when the defect entered llama.cpp
and which ollama releases carry it.

## Origin

**Commit `6eddde06a4f2` — "CUDA: refactor MMQ kernel configuration (#24127)", build b9992.**

The diff, in `ggml/src/ggml-cuda/mmq.cu`:

```diff
-            get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);            // !ids branch
+            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
-        get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);                // ids branch
+        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
```

Both branches were changed identically. That is correct for the `!ids` branch, where `ne11`
really is the quantised buffer's row count, and wrong for the `ids` branch, where the row
count is `ne12*n_expert_used` (named `ne11_flat` a few lines below). The previous
`get_mmq_x_max_host(cc)` was a compute-capability constant — 128 for WMMA/Turing, else 64 —
so the old code always had padding, and the refactor made it zero whenever `ne11 == 1`.

- **last clean build:** b9990 (`259ae1df8b52`)
- **first defective build:** b9992 (`6eddde06a4f2`)
- b9991 is not a tag; the only other commit in the range (`e920c523e3b8`) is Vulkan-only.

## ollama release mapping

`LLAMA_CPP_VERSION` at each release tag of ollama/ollama:

| ollama | llama.cpp | defect |
|---|---|---|
| v0.31.0, v0.31.1 | b9840 | no |
| v0.31.2, v0.32.0, **v0.32.1** | **b9888** | **no — last clean release** |
| **v0.32.2** | **b10069** | **yes — first affected release** |
| v0.32.3, v0.32.4, v0.32.5 | b10091 | yes |
| v0.32.6, v0.32.7 | b10242 | yes |
| v0.32.8, v0.32.9 | b10353 | yes |
| v0.32.10 – v0.32.13 | b10380 | yes |
| master (2026-08-14) | b10434 | yes |

Confirmed at runtime as well as by source: stock `ollama/ollama:0.32.1` (b9888) is clean in
**3/3 cold trials** on sm_120 with `n_ubatch = 2048` and `n_tokens_batch = 2040`, against
`f8def7fe1` (b10353) which faults reliably.

## Consequences

- The gfx1151 ROCm host runs b9888 and is **not affected**. Do not backport `903` to
  `release/0.32.1-dynres`; there is nothing there to fix and the patch will not apply.
- Any fork or deployment pinned at or below b9888 is unaffected.

## Method note — a trap worth avoiding

`raw.githubusercontent.com` does **not** reliably 404 for a non-existent tag. Probing `b10024`
(which is not a tag) returned a 4,694-byte `mmq.cu` from some unrelated revision, whose
content read as "clean" and produced a bisect that pointed at the wrong commit entirely
(`a3e5b96ac5e2`, a Vulkan-adjacent concat change).

llama.cpp does not tag every build number — of b9900..b10079 only 133 tags exist. So:

1. Enumerate real tags first (`git ls-remote --tags`), and bisect over that list.
2. Sanity-check every fetched file before reading a verdict from it. Here, requiring
   `nbytes_src1_q8_1` to be present was enough to reject the bogus fetches.
