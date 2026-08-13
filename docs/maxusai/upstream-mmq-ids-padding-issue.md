# CUDA: MUL_MAT_ID illegal memory access — MMQ ids path sizes tail padding from `ne11` instead of the flattened row count

### Summary

`ggml_cuda_mul_mat_q()` under-allocates `src1_q8_1` in the `ids` (MUL_MAT_ID) branch. The
data term is sized for `ne12 * n_expert_used` rows, but the MMQ tail-padding term is computed
from `ne11`. For MoE gate/up projections the activations are broadcast across experts, so
`ne11 == 1`, and `ggml_cuda_mmq_get_J_max()` then returns **0** — the buffer receives no tail
padding at all, while MMQ processes rows in tiles of up to `J_max = 512` and reads past the
logical end.

The result is an out-of-bounds read. It faults only when it happens to cross an unmapped
page, which is why it presents as intermittent.

### Affected code

`ggml/src/ggml-cuda/mmq.cu`, in `ggml_cuda_mul_mat_q()`, the `ids` branch:

```c
const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
    ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
```

Twelve lines below, the row count the quantise kernel actually writes is named explicitly:

```c
const int64_t ne11_flat = ne12*n_expert_used;
```

`ggml_cuda_mmq_get_J_max()` in `ggml/src/ggml-cuda/mmq.cuh`:

```c
static __host__ int ggml_cuda_mmq_get_J_max(const ggml_type type, const bool fallback, const int cc, const int64_t ne11) {
    int ret = std::min(ne11, int64_t(512));
    ret -= ret % 8;
    for (;ret > 0; ret -= 8) {
        if (ggml_cuda_mmq_get_config(type, ret, fallback, cc).type != GGML_TYPE_COUNT) {
            return ret;
        }
    }
    return ret;
}
```

With `ne11 == 1`: `ret = 1`, then `ret -= 1 % 8` makes `ret = 0`, the loop body never
executes, and the function returns `0`.

The non-broadcast case is under-allocated as well, though less severely — `ffn_down` passes
`ne11 = n_expert_used = 8`, sizing padding for 8 rows rather than `ne12 * n_expert_used`.

By contrast the `!ids` branch a few lines above passes `ne11` correctly, because there `ne11`
*is* the row count of the quantised buffer.

### Observed fault

Instrumented build with a `cudaStreamSynchronize(ctx.stream())` inserted immediately before
the existing `cudaGetLastError()` in `ggml_cuda_compute_forward`, so the error is attributed
to the node that caused it rather than to a later dispatch:

```
op=MUL_MAT_ID name=ffn_moe_gate-3
  dst_ne=[512, 8, 2040, 1]
  src0=blk.3.ffn_gate_exps.weight : q4_K [2048, 512, 256, 1]
  src1=attn_post_norm-3 (reshaped): f32  [2048,   1, 2040, 1]
  src2=ffn_moe_topk-3             : i32  [   8, 2040,    1, 1]
branch=mmq ne0=512 ne1=8 ne2=2040 ne02=256 ne11=1 ne12=2040 type=q4_K
```

```
ggml_cuda_compute_forward: MUL_MAT_ID failed
CUDA error: an illegal memory access was encountered (ggml-cuda.cu:2374)
```

### Environment

- RTX PRO 6000 Blackwell (sm_120), CUDA 13.0, driver-JIT from PTX (`120-virtual`)
- llama.cpp `f8def7fe1`; also reproduced on `cb295bf59` and `b4d6c7d8f`
- MoE model with 256 experts, 8 used, `q4_K` gate/up and `q6_K` down
- CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) — the fault is unaffected by them

### Reproduction

Requires a MUL_MAT_ID call with `ids != nullptr`, `ne11 == 1` (broadcast activations), and a
large `ne12`. In practice: a MoE vision model with an image large enough that its token chunk
arrives in a single ubatch. Concretely, ~2040 image tokens at `-b 2048 -ub 2048`.

Note that upstream's default `n_ubatch` of 512 splits such an image into four chunks, which
is why this shape is rarely produced by llama.cpp's own defaults — it needs `-ub` raised to
roughly the image's token count.

### Why it looks intermittent

The overrun is a fixed size past the end of the allocation, so whether it faults depends on
the pool's history. Measured on the same container:

- cold, 2040-token image, first request: crashes reproducibly
- the identical request after the pool had served a 4080-token image: passes

A green run is therefore not evidence of absence. `test-backend-ops` is likewise not a
reliable oracle here: the over-read lands in padding rows whose results are discarded, so the
NMSE comparison is unaffected — output is byte-identical with and without the fix. The
symptom is the crash only, not incorrect results. `compute-sanitizer --tool memcheck` does
flag it.

### Suggested fix

Size the padding from the row count the buffer actually holds:

```diff
     const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
-        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
+        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12*n_expert_used) * sizeof(block_q8_1_mmq);
```

Verified on the configuration above: cold container, target request first, four consecutive
runs, no fault. The instrumented branch log confirms the same `mmq` branch with identical
shapes still executes, so the fix corrects the path rather than routing around it.

Happy to open a PR if the approach looks right. `ggml_cuda_mmq_get_J_max()` returning 0 for
small `ne11` may also deserve a guard in its own right, independently of this call site.

### Possibly related

- #22867 — same model family, same `find_slot: non-consecutive token position` line
- #18331 — MMQ illegal access at a ubatch boundary on Blackwell, attributed there to nvcc
  codegen; worth re-checking against this, since the workaround
  (`-DCMAKE_CUDA_ARCHITECTURES=89`) would also change pool/allocation behaviour
- #24399
