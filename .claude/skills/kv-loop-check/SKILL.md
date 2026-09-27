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
- **f32 with flash attention on is f16, byte for byte, on CUDA and HIP at b11081** (ADR 0044, SPEC H25).
  `fattn.cu` converts f32 K/V to f16 before its kernels. Never run it as an arm. The one exception is a single
  `ARMS="f16:1 f32:1"` pair after a llama.cpp bump, compared byte for byte.
- **No KV type or attention path reliably turns the known loops into finishes.** Where a loop starts moves with
  the numerical path, in both directions (both hosts, 2026-09-26 and 27).
- **The known loops are prompt-driven.** They come from asking for absolute pixel coordinates without giving the
  image's size (`bbox_contract_real_1img`, `bbox_contract_adv_real`). The same scene in normalized coordinates
  finishes.
- **The suite's think-on is greedy, which is the worst case** (`sampling.py`, `THINK_TEMPERATURE=0`). Production
  sends the model card's sampling.

## Procedure

1. **Confirm it is a loop.** Run `vision-suite/kvloop_read.py` on the capture. It reports `done_reason`, where the
   loop starts (the first 100-line window with under 5% new lines), and the second half's distinct lines. A long
   thinking that finishes is not a loop.
2. **Separate the case from the run's history.** Capture the cell cold with `vision-suite/thinkcap.py`, which
   evicts every model first. Compare it byte for byte with the suite's `think_<tag>_<test>.txt`.
   - **Identical:** the trajectory is the case's own.
   - **Diverges early, with the same `prompt_eval_count`:** the run's state did it, not the setting under test.
     The likely causes are the prompt cache from the previous cell and the parallel slot.
3. **Only if the question is precision, run the matrix.** Use `vision-suite/kvloop.sh`; its default
   `ARMS="f16:1 f16:0 f32:0"` is the whole useful matrix. `q8_0` belongs only as a flash-attention-on control,
   because a quantized V cache needs flash attention. Read the runner's logged `--cache-type-k/v` and
   `--flash-attn` flags. Unset `OLLAMA_FLASH_ATTENTION` means `auto`, which turns flash attention on.
4. **Test the prompt and production's sampling.** Use `vision-suite/promptcap.py`: `size` states the image size,
   and `commit` asks the model to commit to one size estimate. Setting `THINK_TEMPERATURE=1` gives the card's
   sampling. Sampled runs are draws, so report a rate over three or more runs, never one cell.
5. **Write the result into the task doc's section for your host, or comment on the PR.** Give the arms, the
   runner's flags, the onsets and the byte-identities.

## Traps

- `timeout` starts its child in a **new process group**. To stop a `timeout … python3 thinkcap.py` job, kill that
  group as well as the script's.
- **`/tmp` does not survive a reboot.** Keep captures, logs, watchers and helper scripts on durable disk.
- A looping 57,344-token capture on a shared GPU outlives the suite's derived HTTP timeout. `kvloop.sh` sets
  `HTTP_TIMEOUT=9000`.
- Sharing the GPU changes timing, never greedy output. Do not report timing from a shared run.
- `q8_0` contains an underscore. Match arm labels; do not split on `_` (`kvloop_read.py` does this).
