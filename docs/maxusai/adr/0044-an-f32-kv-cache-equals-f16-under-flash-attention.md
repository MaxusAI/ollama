# ADR 0044: an f32 KV cache equals f16 under flash attention, so no test runs it there

- **Status:** accepted 2026-09-27 (the maintainer: record it "so we can confidently drop f32 and save time"). It
  sits beside [ADR 0043](0043-production-runs-an-f16-kv-cache-on-every-platform.md), which makes f16 production's
  KV cache, and it is enforced in the harness by
  [SPEC H25](../spec/vision-harness-reuse.md).
- **Date:** 2026-09-27
- **Deciders:** MaxusAI fork maintainers

## Context

The KV-precision and flash-attention loop test ([kv-precision-think-loops.md](../tasks/kv-precision-think-loops.md),
#387) ran f32 with flash attention on as an arm. The code predicted the result. At b11081, CUDA's and HIP's flash
attention convert an f32 K/V cache to f16 before their kernels run: `ggml/src/ggml-cuda/fattn.cu` sets
`need_f16_K = K->type == GGML_TYPE_F32 …` on the vector kernel, and the tile and MMA kernels always take f16. So
the attention reads the same f16 values it would read from an f16 cache.

**The cast happens earlier, on every backend** (the Metal host on #387, 2026-09-28). llama.cpp's own graph casts an f32
K and V to f16 just before `ggml_flash_attn_ext`, whatever the backend (`build_attn_mha`, `src/llama-graph.cpp`,
b11081). So an f32 cache never reaches a flash-attention kernel, not even Metal's, which has f32 variants.
`sched_reserve`'s node counts show the casts: two per attention layer under f32 with flash attention on, none with it
off. `fattn.cu`'s conversion is then moot.

The measurements agree, byte for byte:

| host | captures | f32 against f16, flash attention on |
|---|---|---|
| gfx1151 (HIP, `amd-server`) | qwen3.6 `bbox_contract_real_1img`, cold, 65536 | thinking byte-identical, 143,475 characters; both loop to the 57,344-token cap |
| CUDA (sm_120, `ai-server`) | gemma4:26b, three cases, on the fold and the 908 build | byte-identical in all six pairs, thinking and answer |
| Metal (llama.cpp's Metal backend, `mlx-metal`) | qwen3.6 and gemma4:26b, two cases each; gemma4's at a pinned batch | byte-identical in all four pairs |

So f32 with flash attention on costs twice the KV memory and bandwidth for identical output. Each such capture
took up to an hour of shared GPU time.

With flash attention off, f32 does change the result, because the attention then runs in f32 through `mul_mat`.
It is not the better setting. On gfx1151 it loops earliest on both looping cases (qwen3.6 from about token 7,390,
gemma4:26b from about 3,000). On CUDA it loops on `multi_3img_anchored`, where f16 with flash attention off
finishes.

## Decision

1. **No KV test runs f32 with flash attention on, on any llama.cpp backend.** It is f16 with flash attention on, at twice the
   memory. A matrix runs f16 with flash attention on (production), f16 with flash attention off, and, when the
   question is precision, f32 with flash attention off.
2. **f32 with flash attention off is a precision diagnostic, not a candidate setting.** It is the most precise
   attention the build has, and it is slower at long context. It did not reduce loops.
3. **Production never runs f32.** ADR 0043 makes it f16.
4. **Scope, and when to re-check.** This holds for CUDA, HIP and Metal at b11081, because the cast is in
   llama.cpp's graph. When the llama.cpp pin moves and `build_attn_mha`'s cast changes, re-take one pair:
   `kvloop.sh` with `ARMS="f16:1 f32:1"` on one case, cold, at a pinned batch, compared byte for byte. That takes
   about 30 minutes. The MLX runner has no KV type to vary (the Metal host on #387).

## Consequences

- `kvloop.sh`'s default arms are `f16:1 f16:0 f32:0`. `f32:1` stays accepted, for the re-check in decision 4.
- A future matrix is a quarter smaller. On the 2026-09-26 run that is one looping capture per case, up to an hour
  each on a shared GPU.
- The loop question itself is answered in the task doc: no KV type or attention path reliably turns these loops
  into finishes, and the prompt is the trigger.
