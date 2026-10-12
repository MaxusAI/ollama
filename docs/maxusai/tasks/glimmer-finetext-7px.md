# TASK: why does muse-glimmer read none of the 7 px fine text, and would a bigger image budget help?

Opened by the Metal host (`macbook-pro-m5-max-128GB/mlx-metal`) on 2026-10-13, for another agent to pick up.
Status: **open, unclaimed.** Context: #463 (onboarding), #468 (an explicit `image_max_tokens` is honoured).

## What was seen

`finetext` (four codes per size tier, exact match, think off, #464's build `0.40.2-dynres-1-g4cfad4a`): every glimmer
tag reads 4/4 at 22, 16 and 12 px, 3–4/4 at 9 px, and **0/4 at 7 px**, on both runners. gemma4:26b and :31b
(nvfp4) read 3/4 at 7 px on the 0.40.2 fold build.

The image budget is not obviously the limit:

| model | `finetext` `prompt_eval_count` | image tokens (approx.) | 7 px |
|---|---|---|---|
| every glimmer tag, MLX and GGUF | 3259 | ~3136 | 0/4 |
| gemma4:31b-nvfp4 | 1173 | ~1120 (gemma4's top rung) | 3/4 |

- `visimgs/finetext.png` is 1568×1568.
- **Glimmer's vision stride is 28** (patch 14, downsample 2), so the card is a 56×56 grid, 3136 tokens. That is the
  image at native resolution.
- **Both runners keep an aspect-preserving grid, never upscale, and cap at 4096 tokens:**
  - MLX: `mlxrunner/model/glimmer/media.go`.
  - llama.cpp: `PROJECTOR_TYPE_MUSE_GLIMMER`, `set_limit_image_tokens(1, 4096)`.
- **So glimmer already sees about 2.8× gemma4's tokens, at native resolution.** A budget above 1120 is already in
  effect, because unset means glimmer's own 4096 on MLX (#468) and GGUF ignores the option. Raising
  `image_max_tokens` cannot add detail unless the image is upscaled first.

## What to do

1. **Confirm the token counts from the runner,** not by arithmetic: the MLX runner's image grid log line and
   llama-server's `image_tokens`.
2. **Upscale on the client.** Send `finetext.png` resized to 1792×1792 (64×64 = 4096, the cap) and to 2× (it then
   hits the cap and is scaled back down). Record 7 px recall per size, `nvfp4` (MLX) and `q8_0` (GGUF, `NUM_BATCH=2048`).
3. **Crop.** Send only the 7 px region at its native size and at 2×. If a 2× crop reads, the limit is pixels per
   glyph (7 px text is a quarter of one 28 px token), not the model's reading.
4. **Check upstream:** the model card on ollama.com and the original model's card, plus llama.cpp and MLX issues for
   the projector. Look for a recommended input resolution, or a tiling or "any-res" mode that ollama does not
   implement.
5. **Compare with gemma4 at the same pixels per token.** Both runners' thinking off, num_ctx 16384, greedy, under
   the host's GPU gate. Never touch :11435.

## Done when

- The record states, with numbers, whether 7 px is out of reach at glimmer's 4096 cap, and whether client-side
  upscaling or cropping recovers it.
- If ollama is missing a preprocessing mode the model expects (tiling, a higher cap), a fix or a finding issue
  exists.
- If not, the README's glimmer row says that 7 px needs a crop.
