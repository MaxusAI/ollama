---
name: kv-loop-check
description: Find out whether a think-on loop (thinking that repeats until the token cap) comes from the KV cache, the attention path, the run's history, or the prompt, using cold captures and the fork's KV x flash-attention harness. Use when a think-on cell hits its cap or never converges on the ladder, when someone asks whether q8_0, f16, f32 or flash attention causes or cures a loop, when planning a KV-precision test on CUDA or ROCm, or before changing OLLAMA_KV_CACHE_TYPE anywhere. NOT for think-off regressions or throughput (SPEC H22), and NOT for MLX, whose runner has no KV-type or attention knob.
---

# Think-on loops: KV precision, attention path, history, or prompt?

This skill owns the **procedure**. The evidence lives in
`docs/maxusai/tasks/kv-precision-think-loops.md`, and the decisions in ADR 0043, ADR 0044 and SPEC H25.

## Already settled: do not re-measure

- **Production runs an f16 KV cache on every platform** (ADR 0043). gfx1151 ran `q8_0` by accident from
  2026-08-08 to 2026-09-26, so any gfx1151 think-on result from that window carries that caveat.
- **f32 with flash attention on is f16, byte for byte, on CUDA, HIP and Metal at b11081** (ADR 0044, SPEC H25).
  llama.cpp's graph casts f32 K/V to f16 before flash attention, on every backend (`build_attn_mha`). Never run it as an arm. The one exception is a single
  `ARMS="f16:1 f32:1"` pair after a llama.cpp bump, compared byte for byte.
- **No KV type or attention path reliably turns the known loops into finishes.** Where a loop starts moves with
  the numerical path, in both directions (both hosts, 2026-09-26 and 27). It can go against f16 too: on gfx1151,
  gemma4:26b's `multi_3img_anchored` finishes under `q8_0` and loops under f16.
- **Most known loops are prompt-driven.** They come from asking for absolute pixel coordinates without giving the
  image's size (`bbox_contract_real_1img`, `bbox_contract_adv_real`). The same scene in normalized coordinates
  usually finishes, but not always (next bullet).
- **Some loops had no known trap sentence.** gemma4:26b has looped twice on the same scene in norm-1000, with no
  known trap in the prompt:
  - On CUDA, the 908 build with flash attention on and f16 loops `bbox_contract_box2d_1img` from about token 4,247
    (the CUDA table in `docs/maxusai/tasks/kv-precision-think-loops.md`).
  - On gfx1151, `bbox_contract_anchored_1img` loops under greedy f16 (2026-09-29).
    - A cold capture repeats the suite's thinking byte for byte.
    - f16 and `q8_0` part 662 characters in, where they word the corner of the shape labelled ANCHOR differently.
    - `q8_0` finishes, and so does f16 with flash attention off (2,807 tokens).
    - Under the card's sampling, 0 of 10 f16 draws loop (below about 26% at 95%).

  Both prompts are built on `_BBOX_PLACEMENT_HEAD`, whose "the others are distractors and must be ignored" is false
  when one image is sent. On gfx1151's case, dropping it (`promptcap.py`'s `nodistract`) finishes under greedy f16,
  but so do a true rewording of it (`truedistract`) and a neutral edit elsewhere (`neutral`), and CUDA's
  `box2d_1img` does the same under the same three edits. So it is not singled out on either host: every prompt
  edit tips both greedy trajectories.
  - On gfx1151, `q8_0` and flash attention off tip it too.
  - On CUDA, the fold's FA tiling tips it too. f32 with flash attention on, the flow and the slot count leave the
    loop byte-identical.

  gfx1151's cold repeat, KV sensitivity and sampled finishes also fit the trap cases, so they do not tell a trap
  from a non-trap. Only the prompt text does, and only against a neutral-edit control. Before calling a greedy loop
  a regression, measure it with the card's sampling and report a rate, k of n.
- **The suite's think-on is greedy, which is the worst case** (`sampling.py`, `THINK_TEMPERATURE=0`). Production
  sends the model card's sampling. On the pixel-coordinate case, all 6 card-sampled runs finish (2026-09-27).
- **The KV type moves greedy think-on scores more than a fold's code change, in both directions.** In the v0.34.4
  fold's protocol on gfx1151 it moved the quality of 20 of 25 qwen3.6 tests, where the fold's structured-output
  change moved 4 (2026-09-27). Compare arms only at one KV type. `vision-suite/cmp_scored.py` shows which quality
  fields moved and which way.
- **A prompt that asks for something the model cannot see is a loop trigger.** "Give the size YOU used" after an
  invisible internal resize loops gemma4:26b and qwen3.6. With it replaced, gemma4 answers correctly. qwen3.6
  answers in 0–1000 whatever the prompt says, so pin norm-1000 (SPEC C1).

## Procedure

1. **Confirm it is a loop.** Run `vision-suite/kvloop_read.py` on the capture. It reports `done_reason`, where the
   loop starts (the first 100-line window with under 5% new lines), and the second half's distinct lines. A long
   thinking that finishes is not a loop.
2. **Separate the case from the run's history.** Capture the cell cold with `vision-suite/thinkcap.py`, which
   evicts every model first. Compare it byte for byte with the suite's `think_<tag>_<test>.txt`.
   - **Identical:** the trajectory is the case's own.
   - **Diverges early, with the same `prompt_eval_count`:** the run's state did it, not the setting under test.
     The likely causes are the prompt cache from the previous cell and the parallel slot. Since
     2026-10-03 harness requests turn llama-server's host-RAM prompt cache off
     (`req_prompt_cache_ram: 0` in the scores). That leaves the slot's own reuse of the previous
     prompt's prefix.
3. **Only if the question is precision, run the matrix.** Use `vision-suite/kvloop.sh`; its default
   `ARMS="f16:1 f16:0 f32:0"` is the whole useful matrix. `q8_0` belongs only as a flash-attention-on control,
   because a quantized V cache needs flash attention. Read the runner's logged `--cache-type-k/v` and
   `--flash-attn` flags. Unset `OLLAMA_FLASH_ATTENTION` means `auto`, which turns flash attention on.
4. **Test the prompt and production's sampling.** Use `vision-suite/promptcap.py`: `size` states the image size,
   and `commit` asks the model to commit to one size estimate. Both rewrite the unanswerable sentence in the two
   cases that were studied: `bbox_contract_real_1img`, and `multi_3img_anchored`'s calibration box. `multi_3img` is
   the same prompt without that paragraph, so it is the control. `bbox_contract`, `bbox_contract_multi` and
   `bbox_contract_reasoning` ask for the size in other words, and both variants refuse them. `nodistract` drops
   `_BBOX_PLACEMENT_HEAD`'s "the others are distractors and must be ignored", which is false on the single-image
   arms built on it (`anchored_1img`, `box2d_1img`, `positional_1img`). `truedistract` rewords that sentence
   truthfully and `neutral` drops a different, true sentence: run them as controls, because a greedy trajectory can
   tip on any edit, and a sentence is singled out only if the neutral edit does not also finish. Setting
   `THINK_TEMPERATURE=1` gives the card's sampling. Sampled runs are draws, so report a rate over three or more
   runs, never one cell.
5. **Write the result into the task doc's section for your host, or comment on the PR.** Give the arms, the
   runner's flags, the onsets and the byte-identities.

## Traps

- `timeout` starts its child in a **new process group**. To stop a `timeout … python3 thinkcap.py` job, kill that
  group as well as the script's.
- **`/tmp` does not survive a reboot.** Keep captures, logs, watchers and helper scripts on durable disk.
- A looping 57,344-token capture on a shared GPU outlives the suite's derived HTTP timeout. `kvloop.sh` sets
  `HTTP_TIMEOUT=9000`.
- **Sharing the GPU can change greedy output, through the batch.** Do not report timing from a shared run either.
  - The fork's automatic batch (`automaticGenerationBatch`, `server/sched.go`) picks llama-server's `-b/-ub` from the
    memory free at each launch. A neighbour's memory use can lower it: 2048 → 1024 → 512.
  - **CUDA with flash attention off is the exception.** There the batch is always 512 (256 above 4096 context on a
    GPU of 8 GiB or less), set before the image-chunk floor and the memory check (upstream's rule,
    ollama/ollama#16353). So a CUDA flash-attention-off arm runs gemma4's images in pieces even on a quiet GPU. HIP
    and Metal keep the automatic batch.
  - gemma4's greedy output moves with the batch, in both directions (the Metal host on #387: `real_1img` loops at
    512 and answers at 2048; `multi_3img_anchored` the reverse). Its SWA cache and its image chunking both depend
    on it. qwen3.6's does not.
  - Before comparing gemma4 captures across arms or hosts, read each launch's `-b/-ub` and look for "images decode
    in pieces". To take the batch out of the comparison, pin it: `NUM_BATCH=2048` (the suite's `-b/-ub` pin).
- `q8_0` contains an underscore. Match arm labels; do not split on `_` (`kvloop_read.py` does this).
