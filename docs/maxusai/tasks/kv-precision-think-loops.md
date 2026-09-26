# TASK: KV precision and the attention path against the think-on loops

**Question (the maintainer, 2026-09-26):** does the KV cache's precision decide the think-on loops, and does the
flash-attention path? Every host runs the same cold captures on the cases that loop on it, and adds its results
here or as a comment on the pull request that opened this file. Production's KV cache is f16 on every platform
([ADR 0043](../adr/0043-production-runs-an-f16-kv-cache-on-every-platform.md)). This task measures whether f16
is enough, and whether the attention path matters.

## What is compared

Each looping case is captured **cold**, with every model evicted first, by the suite's own request
([`thinkcap.py`](../vision-suite/thinkcap.py)). The capture runs on the host's fold image, in the single-pass
flow, and in production's environment with only two knobs changed:

| KV cache \ flash attention | on | off |
|---|---|---|
| `q8_0` | the control, where it was the environment | does not exist: a quantized V cache needs flash attention |
| f16 | production (ADR 0043) | the flash-attention kernels against `mul_mat`, at the same storage |
| f32 | expected to equal f16 on CUDA and HIP (see below) | the most precise attention the build has |

- **Why f32 needs flash attention off.** At b11081, CUDA's and HIP's flash attention convert an f32 K/V cache to
  f16 before their kernels run (`ggml/src/ggml-cuda/fattn.cu`: `need_f16_K = K->type == GGML_TYPE_F32 …`; the
  tile and MMA kernels always take f16). With flash attention on, f32 storage should reproduce f16 exactly, at
  twice the memory. Only with flash attention off does the attention itself run in f32. Metal's llama.cpp
  backend may differ.
- **Budget.** `num_ctx` is 65536, so the capture has 57344 tokens (`thinkcap.py` sets `num_predict = num_ctx -
  8192`). ADR 0005 found qwen3.6's trajectories token-identical across `num_ctx`, so budgets compare as token
  counts.
- **Readout** ([`kvloop_read.py`](../vision-suite/kvloop_read.py)):
  - how the capture ended (`done_reason` and tokens);
  - the thinking's repetition profile: distinct lines in the whole and in the second half, and the line
    repeated most;
  - the answer, scored by the suite's own scorer.

  A loop shows as a second half with few distinct lines. For the adversarial arms (`adv_*`), read `hits_anchor`
  and not `contract_followed`, as the suite notes.
- **Loop onset.** `kvloop_read.py` also reports where the loop starts: the first 100-line window in which under 5%
  of the lines are new to the thinking. This finds cycles of any period. It gives the line and an estimate of the
  token, scaled by the capture's characters per token.
- **Flags.** `OLLAMA_FLASH_ATTENTION=0` makes the fork pass `--flash-attn off`. If it is unset, the fork passes
  `auto`, which enables flash attention on these GPUs. Record the runner's `--cache-type-k`, `--cache-type-v` and
  `--flash-attn` flags with each capture; [`kvloop.sh`](../vision-suite/kvloop.sh) logs them.

## How to run it

On a docker host (ROCm or CUDA), `kvloop.sh` starts one container per arm and captures every case:

```
IMG=<the fold image> STORE=<model store> GPU_ARGS="--gpus all" \
  ARMS="f16:1 f16:0 f32:0 f32:1" \
  CASES="gemma4:26b-a4b-it-q4_K_M|bbox_contract_real_1img" \
  OUT=kvloop-<host> ./kvloop.sh
python3 kvloop_read.py kvloop-<host>/*.json
```

On ROCm, `GPU_ARGS` is `--device /dev/kfd --device /dev/dri` plus the host's numeric `--group-add` IDs for
`video` and `render`. On a native host, start the server once for each arm with `OLLAMA_KV_CACHE_TYPE` and
`OLLAMA_FLASH_ATTENTION` set. Then run this for each case:

```
python3 thinkcap.py <host> <model> <test> 65536 <out.json>
```

## gfx1151 (ROCm, `amd-server`)

The runs use the fold image `0.34.3-dynres-5-g29ae523` (b11081), single pass, `OLLAMA_NUM_PARALLEL=2`. The first
column is the aligned protocol's (#375), which ran with production's `q8_0` at the time. The other columns are cold
captures.

| case | `q8_0`, FA on, in the protocol | `q8_0`, FA on, cold | f16, FA on | f16, FA off | f32, FA off | f32, FA on |
|---|---|---|---|---|---|---|
| qwen3.6 `bbox_contract_real_1img` | never finishes at 131072; second half 35/2282 lines distinct | loops: all 24576 tokens at 32768; second half 49/436 | **loops**: all 57344 tokens, no answer; second half 77/1647 | queued | queued | queued |
| qwen3.6 `bbox_contract_adv_real` | never finishes at 131072; second half 27/3165 | **finishes**: 17,626 tokens, valid JSON, 6/6 labels | **finishes**: 12,120 tokens, valid JSON, 6/6 labels | — | — | — |
| gemma4:26b `bbox_contract_real_1img` | never finishes at 131072, in both flows | loops: all 24576 tokens at 32768; second half 10/381 | **loops**: all 57344 tokens, no answer; second half 26/1224 | queued | queued | — |

**Captured cold, f16 turns no loop into a finish. It delays the loops, and it shortens the thinking in the case that
finishes.** Estimated token at which the loop starts:

| case | `q8_0` | f16, FA on |
|---|---|---|
| qwen3.6 `bbox_contract_real_1img` | about 14,650 (cold) and 14,890 (in the protocol) | about 23,000 |
| qwen3.6 `bbox_contract_adv_real` | about 8,750 in the protocol; cold, no loop, finishes at 17,626 | no loop; finishes at 12,120 |
| gemma4:26b `bbox_contract_real_1img` | about 3,290 (cold) | about 4,670 |

**Cold captures isolate the KV type. The protocol's cells can also carry the run's history.**

- qwen3.6 `real_1img`'s cold `q8_0` capture at 32768 is byte-identical to the protocol's thinking at 131072 for its
  whole length, 58,820 characters. That trajectory depends on neither history nor `num_ctx`, which matches ADR 0005.
  Its cold f16 capture diverges from the cold `q8_0` one 246 characters in ("Single JSON object" against "A single
  JSON object").
- qwen3.6 `adv_real` is different. Its cold `q8_0` capture takes the same 5,930 prompt tokens as the protocol's
  cell, and finishes. The protocol's thinking diverges from it 369 characters in, and never finishes. So the
  protocol's loop on this case came from the run's state and not from the KV type. The likeliest causes, which
  are inferred and not tested, are the prompt cache reused from the previous cell (the same first image and
  prompt head) and the parallel slot the request landed on.

**So f16 with flash attention on turns none of the three cold cases from a loop into a finish.** qwen3.6 `adv_real`
finishes under both KV types, 5,500 tokens sooner with f16. qwen3.6 `real_1img` and gemma4:26b `real_1img` loop under
both, and f16 moves the start of the loop later. gemma4's thinking locks up after its first quarter (157, 20, 18
and 26 distinct lines out of 612 per quarter), and repeats "Wait, I'm still getting the same numbers. Let me
re-examine the image." 77 times.

**What the looping thinking goes over, in both qwen3.6 cases.** The prompts ask for pixel coordinates without
giving the image's size, and the model keeps re-deciding it: "I will assume W=1920, H=1080 … Maybe the image is
smaller … Let's assume W=1200, H=675". In `adv_real` it commits to one size and finishes, under both KV types. Both
answers declare `ref_size` [1000, 600] against the true 1920×1080. Their `hits_anchor` is 0, and 3 of 6 boxes hit
only in the best-fitting dialect, norm-1000.

## CUDA (`ai-server/mlx-cuda`)

**Queued on the CUDA host, after its drafting probe** (#387, 2026-09-26). Its production and gate runs were f16 already
(#375). The plan:

- **Cases:** gemma4:26b-a4b-it-q4_K_M, with three cases:
  - `multi_3img_anchored`, which loops on the fold and finishes with 908;
  - `bbox_contract_real_1img`, the case shared across hosts;
  - `bbox_contract_box2d_1img`, the 908 build's one loop.
- **Builds:** each case runs on both builds, with `ce8caa6e6`'s tiling (the fold as shipped) and without it (the 908
  image).
- **Arms:** `f16:1 f16:0 f32:0 f32:1`. `f16:1` stays in as the run's own control, because the loop-rate table on
  #375 ran in the suite at the default `OLLAMA_NUM_PARALLEL`, and `kvloop.sh` captures cold at 2.
- **Deploy source (ADR 0043, decision 1):** production does not set the variable today and runs the f16 default
  (12 of 12 KV allocations). The v0.34.4 deploy sets `OLLAMA_KV_CACHE_TYPE=f16` explicitly. It refuses if the live
  container carries a different value.

## Metal (`mlx-metal`)

**On MLX, neither knob exists** (the Metal host on #387, from the fold's code at `29ae52351`):

- `OLLAMA_KV_CACHE_TYPE` and `kv_cache_type` are resolved only for llama-server (`resolveKVCacheType`, called from
  `NewLlamaServer`). The MLX runner's `KVCache` allocates in the dtype of the model's own projections
  (`mlxrunner/cache/kvcache.go`).
- Every MLX text model's attention goes through `mlx.FastScaledDotProductAttention`, MLX's fused kernel
  (`mlxrunner/nn/sdpa.go`), and no variable switches it.
- On MLX, drafting and request history are what move the loops (#375). Varying the KV precision or the attention
  path there would take a code change.

**GGUF on llama.cpp's Metal backend does apply.** The arms can run there. The GPU time is the maintainer's call,
because the host's protocol campaign is using it.

**Deploy source (ADR 0043, decision 1):** production is a launchd agent. It sets `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`
and no `OLLAMA_KV_CACHE_TYPE`, and it could hold one. Setting f16 there is a production change for the maintainer,
and it affects only the GGUF models the server runs.
