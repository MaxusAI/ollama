# qwen2.5vl:3b — one image yields `???` garbage and permanently poisons the runner until reload

**Status: OPEN bug report (measured 2026-08-26). Reproduces on STOCK `ollama/ollama:0.32.15` ⇒ upstream ollama bug, not a fork regression — should also be reported upstream.**

## Summary

A specific, ordinary-looking corpus image makes **qwen2.5vl:3b in every quant (q4_K_M, q8_0, fp16)** return `done_reason: null` with `???????…` garbage content — and after that one request, **the resident runner returns the same garbage for every subsequent request** (any image, any prompt) until the model is reloaded (`ollama stop <tag>`). The 7B of the same family (3584-embed) processes the identical image cleanly across a full 2,496-request production run; 32B untested but shares the 7B embed width.

The permanent-poisoning aspect is the dangerous part: one bad image silently invalidates every later answer from the resident model.

## Minimal repro

One `/api/chat` request:

- model: `qwen2.5vl:3b-q4_K_M` (or `3b-q8_0` / `3b-fp16`)
- messages: any system + user text, `images: [<the poison image>]`
- options: `{"num_ctx": 8192, "temperature": 0.0, "num_predict": 250}`

Poison image: md5 `02c9d7e1563a7c6089f688ddff8ad590`, RGB PNG 756×1008 (a photograph from a private customer corpus; held on the CUDA box and available privately on request — deliberately not committed and not path-referenced here). A **pixel-identical lossless PNG re-save still triggers it**, so the trigger is pixel content, not container/chunks/ICC.

Response: `done_reason: null`, `content: "???????????????????????????????"`. Every request after it on the same resident instance — including known-good images — returns the same garbage until `ollama stop`.

## Evidence matrix (fresh instance per test; H = healthy, X = garbage)

| Build | Model | Test | Pattern |
|---|---|---|---|
| sync-0.32.15 (vsuite, `0.32.14-dynres-108-g76918a7`, :11502) | 3b-q4_K_M | 10 ordinary images | `HHHHHHHHHH` |
| sync-0.32.15 | 3b-q4_K_M | production sequence, poison at #4 | `HHHXXX` |
| sync-0.32.15 | 3b-q4_K_M | poison FIRST, then known-good | `XX` |
| sync-0.32.15 | 3b-q8_0 | production sequence | `HHHXXX` |
| sync-0.32.15 | 3b-fp16 | poison first, then good | `XX` |
| `0.32.14-rc0-dynres-0-ga5d6590` (:11497) | 3b-q4_K_M | poison first, then good | `XX` |
| **STOCK `ollama/ollama:0.32.15`** (image `38861297e420`, same registry blob `e9758e589d44…`, fresh pull, throwaway container) | 3b-q4_K_M | poison first, then good | `XX` |
| sync-0.32.15 | 7b-q4_K_M | full 2,496-request run incl. the poison image | clean |

Quant-independent (q4_K_M = q8_0 = fp16) ⇒ the shared component is the 3B (2048-embed) mmproj/merger path. Runner log at load:

```
handle_qwen25vl_clip: detected Ollama-format qwen25vl GGUF used as mmproj; translating
load_hparams: projector:          qwen2.5vl_merger
```

Suspicion: the projector emits NaN/garbage embeddings for this input and the runner slot never recovers.

## Server launch flags (vsuite, from `docker logs vsuite`)

```
/usr/lib/ollama/llama-server --model /root/.ollama/models/blobs/sha256-a99b7f83… \
  --port … --host 127.0.0.1 --no-webui --offline -c 8192 -np 1 --log-verbosity 4 \
  --no-jinja --chat-template chatml --mmproj /root/.ollama/models/blobs/sha256-a99b7f83… \
  --image-min-tokens 1024 --flash-attn auto -b 1024 -ub 1024 --split-mode none \
  --main-gpu 0 --context-shift --keep 4
```

(The stock build's flags differ; it fails identically, so the flags are not the trigger.)

## Downstream impact

- Experiment `00017.8`'s qwen2.5vl-3B ladder rungs are blocked; two 2,496-request arms produced garbage before the pattern was isolated. Full diagnosis record lives in the private evaluation workspace, under the experiment's `_quarantine_broken_serving/` notes.
- The consuming harness now aborts after 10 consecutive unhealthy responses (`done_reason` not stop/length, or `???` content) — recommended defense for any sustained-inference client until this is fixed.

## Next steps

1. Reproduce with the attached image against upstream `ollama/ollama` main and file upstream.
2. Bisect the 3B mmproj path (`handle_qwen25vl_clip` translation vs the projector weights) for NaN emission on this input.
3. Consider a runner-level guard: detect degenerate decode (all-`?` / unknown-token loops) and recycle the slot instead of serving poisoned state.

## Root-cause localization (2026-08-26 follow-up)

Evidence accumulated in the PR #214 comment trail, promoted here as the current best
statement of the fault:

**The one fp16 stage common to every failing CUDA config and absent from the healthy
CPU path is the f16-weight matmuls of the vision tower/merger — fp16-accumulate GEMM
on CUDA vs fp32 accumulate on CPU.**

Supporting matrix (full tables and logs in the PR comments):

- **CPU-only serving is HEALTHY** (`NVIDIA_VISIBLE_DEVICES=void`, `size_vram` verified
  0.0): the poison image gets sensible labels. CUDA garbles on **both** Blackwell and
  Turing; the 7B is healthy everywhere → arch-independent within CUDA, model-specific.
- **Every runtime knob is ruled out**: `--flash-attn` on AND off (Blackwell FA-off
  still garbles), KV cache `f32` and `bf16`, `GGML_CUDA_FORCE_MMQ=1` — every CUDA
  config that serves at all produces garbage on the poison request. The fault is in
  vision-encoder compute **shared by the FA and non-FA attention paths**, upstream of
  the KV cache. (clip.cpp at pin `b10488` already sets `GGML_PREC_F32` on vision
  flash-attn and ships the non-FA KQ F32 override commented out — attention precision
  is not the (only) overflow site.)
- **Separable defect**: the permanent slot poisoning is FA-linked on Blackwell (FA on
  → sticky `XX` until reload; FA off → next request healthy, `HXH`; Turing recovers
  even with FA on).
- **Known-good implementation on the SAME blob and SAME GPUs**: `ollama/ollama:0.7.1`
  (Go-engine qwen25vl vision path) = `HHH` on Turing and Blackwell, fully GPU-resident
  (`size_vram` verified; its CUDA payload compiles sm_120), pixel-exact preprocessing
  (980 measured vs 972 expected image tokens) — and it describes the poison image
  sensibly.
- **Regression boundary**: `9db4bdba` "runner: Remove CGO engines, use llama-server
  exclusively for GGML models (#16031)" (2026-05-29) deletes the healthy Go
  implementation and routes qwen25vl to llama-server/clip.cpp. Last release without
  it: `v0.24.0`; first with it: `v0.30.0`. Release-level A/B pending.

Fix candidates: force F32 precision on the clip graph's f16-weight mm ops (the
`build_mm` family, including restoring the commented-out KQ override), or ship the
vision-tower/mmproj tensors as f32/bf16 in the GGUF so the CUDA GEMMs leave the
fp16-accumulate path. Consistent with the widely observed Qwen2.5-VL fp16
activation-overflow behavior (fp16 fine-tuning garbles, bf16 is clean).

Status of the "Next steps" above: (1) **done** — reproduced on stock
`ollama/ollama:0.32.15`; upstream prior art linked in the comments (ollama#14170,
ollama#17687, llama.cpp#23608). (2) **superseded** by this localization. (3) still
recommended as a runner-level defense.

### Correction (2026-08-26, later the same day): 0.7.1 is NOT a clean implementation — trigger sets are disjoint, the class exists in both engines

Running the full production val fold through `ollama/ollama:0.7.1` falsified the
"known-good implementation" claim above. Measured:

| Image (md5, shape) | clip.cpp path (0.32.x) | Go engine (0.7.1) |
|---|---|---|
| `02c9d7e1…` (756×1008) — the original poison | **X** (`?`×31) | H (sensible answer) |
| `04431b0d…` (1288×616) — newly found | H (sensible answer) | **X** (`!`×31) |
| ordinary corpus images | H | H |

`04431b0d…` garbles a **freshly reloaded** 0.7.1 slot on request #1 and poisons it for
every later request (verified after `ollama stop`), i.e. an image-specific trigger with
the exact same signature — degenerate single-glyph decode (`!` here vs `?` there),
`done_reason: null`, sticky slot until reload. The cross-probe on a fresh 0.32.15
container confirms `04431b0d…` is healthy on the clip path.

Interpretation: the overflow class is **latent in the shared CUDA fp16 vision compute of
both implementations**; each graph's op order/precision decides *which* activation
patterns cross the fp16 cliff, so each implementation has its own (disjoint) poison set.
This strengthens the f16-weight-matmul localization above, and narrows the fix guidance:
a clip.cpp-only precision patch would merely move the trigger set — the durable fix is
keeping the vision tower/merger GEMMs out of fp16 accumulation (F32 prec on the mm ops,
or f32/bf16 vision tensors in the GGUF) in whichever engine serves. The release-level
A/B (`0.24.0` vs `0.30.0`) accordingly now reads as a per-image trigger-set boundary,
not as the introduction point of the class.

## Release matrix and workaround (2026-08-26, capstone)

Both trigger images probed per release, fresh container per cell, GPU residency
verified via `/api/ps` (the mmproj-on-CPU rows are fit-check behavior, not config):

| Serving | Vision runs on | `02c9d7e1…` | `04431b0d…` |
|---|---|---|---|
| 0.7.1 (Go engine, full GPU) | GPU | H | **X** |
| 0.24.0 (Go engine, full GPU, 12.14/12.14 GB) | GPU | H | H |
| 0.30.0 (first llama-server release), roomy GPU | GPU | **X** | H |
| 0.30.0, crowded GPU (fit check → `--no-mmproj-offload`) | CPU | H | H |
| 0.32.15 (fork and stock), any CUDA GPU | GPU | **X** | H |
| 0.32.15 with `options.num_gpu = 36` (of 37 layers) | CPU | H | H |
| any build, CPU-only | CPU | H | H |

Conclusions:

1. **The clip-path trigger set has been present since the very first llama-server
   release (0.30.0)** and is stable through 0.32.15. The Go engine had a different
   trigger set at 0.7.1; by 0.24.0 neither known trigger fires there (its own unknown
   set is not excluded — per-implementation sets are the rule).
2. **mmproj offload is memory-conditional** (`shouldDisableMMProjOffload`: cpu-only /
   partial-text-offload / projector+1 GiB fit), so poison exposure varies with free
   VRAM at load time — the same model on the same box can be healthy or poisoned
   depending on what else is resident. This explains intermittent reports.
3. **Operational workaround on current builds**: request `options.num_gpu` one below
   the model's layer count (36 for the 3B) → `partial-text-offload` → llama-server
   gets `--no-mmproj-offload` → the vision path runs on CPU in fp32 and the class
   cannot fire, while the LM stays on GPU. Verified `HHHH` on both triggers.
4. Persistence half of the defect: 0.30.0 (no `--context-shift --keep 4`) recovers
   after a poison request; 0.32.x (with them) stays poisoned until reload — the
   prefix cache appears to retain the NaN state. Worth splitting into its own fix.

## Mechanism confirmed + runtime kill-switch (2026-08-26, Glenn's find)

Glenn asked whether `GGML_CUDA_CUBLAS_COMPUTE_TYPE` had been tried — it had not, and
it exists in the pinned llama.cpp (`b10488` `ggml/src/ggml-cuda/ggml-cuda.cu`), a
runtime override of the cuBLAS mul_mat compute type (`f32|bf16|f16|auto`; default for
f16 weights on fast-fp16 hardware = `CUBLAS_COMPUTE_16F`, i.e. fp16 accumulate).
Measured on the current stack (sync-0.32.15), GPU0, fully GPU-resident, same axis:

| `GGML_CUDA_CUBLAS_COMPUTE_TYPE` | `02c9d7e1…` | `04431b0d…` | slot after |
|---|---|---|---|
| unset (auto → f16 accumulate) | X | H | sticky |
| **`f32`** | **H** | H | clean |
| **`bf16`** | **H** | H | clean |
| `f16` (explicit positive control) | X | — | sticky |

**Root cause confirmed: fp16 accumulation in the cuBLAS GEMMs.** Flipping the
accumulate type heals the poison image at full GPU residency with no CPU cost —
`bf16` is likely perf-neutral (tensor cores, fp32 accumulate), `f32` the maximally
conservative choice. Fix guidance for upstream sharpens to: default the multimodal
(vision tower/merger) matmuls to ≥fp32 accumulation, or at minimum document this env.

Correction to the earlier knob matrix: the `GGML_CUDA_FORCE_MMQ=1` row was an inert
env — neither `GGML_CUDA_FORCE_MMQ` nor `GGML_CUDA_FORCE_CUBLAS` exists in `b10488`'s
runtime env set — so that row duplicated the control, proving nothing about MMQ.

## What makes a poison image a poison — measured (2026-08-26)

Hooked every stage of the HF `Qwen2.5-VL-3B-Instruct` vision tower in bf16
(fp32-like range, so true magnitudes survive) and recorded max |activation| for a
known-good corpus image vs both trigger images
(script: `measure_qwen25vl3b_vision_activations.py`, same smart-resized pixels the
serving path consumes):

| stage | good | `02c9d7e1…` | `04431b0d…` |
|---|---|---|---|
| `blocks.31.mlp` (final block) | 37,376 | **55,808** | **50,688** |
| mid-tower plateau (blocks 17–27) | ~13,000 | ~12,200 | ~13,800 |
| fp16 max finite | 65,504 | 0.85× | 0.77× |

The 3B vision tower has a massive-activation structure in its **final block**: even
ordinary images drive it to ~0.5–0.6× of fp16's ceiling. A "poison" image is simply
one whose content pushes those outlier channels ~50% higher — and while the *stored*
values still fit fp16, the **partial sums inside the next fp16-accumulate GEMM**
(the block-31 MLP projection / the merger consuming it) transiently exceed 65,504
before cancellation, overflow to inf, and cascade to NaN. Hence: deterministic
(same pixels → same activations → same overflow), rare-but-inevitable across a large
corpus (the margin between 37K "fine" and 56K "poison" is thin), implementation-
specific trigger sets (different GEMM tilings hit different partial-sum peaks), and
catastrophic for fine-tunes (any checkpoint that shifts these outlier scales up —
typhoon-ocr, fp16 bitsandbytes training — moves *every* input over the cliff).
bf16/f32 accumulation is immune because the range ceiling moves from 6.5×10⁴ to
~3.4×10³⁸.

## Shareable synthetic repro + the size axis (2026-08-26, capstone 2)

Earlier we reported that synthetic triggers "don't transfer". That was **wrong in one
important respect**, and the correction is the most useful result in this document:
adversarially-optimized noise does not transfer, but **ordinary high-contrast patterns
rendered at a large working size do**.

`make_poison_repro_image.py` (committed alongside) writes a 56 px black/white
checkerboard at 1350x1800. On **stock `ollama/ollama:0.32.9`** with **stock
`qwen2.5vl:3b-q4_K_M`**, fully GPU-resident:

| serving | result |
|---|---|
| default (fp16 cuBLAS accumulate) | **`?` x31, `done_reason: null`** (deterministic, every call) |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | healthy — describes the checkerboard |

No corpus image, no client data, no fine-tune required: three lines of PIL reproduce
the bug end to end. The committed PNG is the exact file we tested.

### Image SIZE is a first-class trigger axis

Measured max |activation| at `blocks.31.mlp` (bf16 HF tower, fp16 ceiling 65,504):

| image | size | max abs | vs fp16 | serving (0.32.9, qwen 3B) |
|---|---|---|---|---|
| corpus poison `04431b0d` | 1288x616 | 50,688 | 0.77x | H on 0.32.x (X on 0.7.1) |
| same, upscaled longest->1800 | 1800x860 | **66,048** | **1.01x** | **X** |
| same, tiled 2x2 -> 1800 | 1800x860 | **70,144** | **1.07x** | **X** |
| corpus poison `02c9d7e1` | 756x1008 | 55,808 | 0.85x | X |
| same, upscaled -> 1800 | 1350x1800 | 62,208 | 0.95x | H |
| ordinary corpus image | 1008x756 | 37,376 | 0.57x | H |
| same, upscaled -> 1800 | 1800x1350 | 59,392 | 0.91x | H |
| synthetic checkerboard p56 | 1350x1800 | 69,120 | 1.06x | **X** |
| synthetic checkerboard p14/p28, rings p5 | 1350x1800 | 50,432-65,280 | 0.77-1.00x | **X** |

Two mechanisms compound, and both point the same way:

1. **Upscaling raises the activation peaks** (~1.6x from 1008 -> 1800 px on the same
   content) — an image that is safe at one working size becomes a trigger at another.
   This retires the idea of a fixed "poison image": the same pixels are safe or fatal
   depending on the size the client sends.
2. **Longer visual sequences lengthen the GEMM reductions**, so partial sums overflow
   even when the *stored* maxima stay under the ceiling (the synthetic rings at 0.77x
   still garble at 3,072 merged tokens, while a natural image at 0.95x does not).

Practical consequence for clients: **the bigger the image, the more exposed you are.**
Anything that renders documents at 1800-2048 px (olmOCR-style pipelines, including
`typhoon_ocr`'s `ocr_document(target_image_dim=1800)`) sits in the worst part of this
space.

### Cross-model note

`scb10x/typhoon-ocr1.5-3b` (upstream ollama#17687, same 31-glyph signature with `@`)
stayed healthy on every one of these images across 0.7.1/0.24.0/0.30.0/0.32.9/0.32.15/
0.33.0, photos and documents alike, including the 1800 px pipeline reproduced from its
own package source. Its fine-tuned vision weights carry different activation scales, so
its trigger set is disjoint from the base model's — the same per-implementation pattern
seen between ollama's Go and clip engines. Finding their specific trigger would need
that checkpoint's HF weights and the same measurement sweep.
