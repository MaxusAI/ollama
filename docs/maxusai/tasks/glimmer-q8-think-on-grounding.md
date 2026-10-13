# TASK: why does muse-glimmer's Q8_0 ground worse with thinking on?

Opened by the Metal host (`macbook-pro-m5-max-128GB/mlx-metal`) on 2026-10-13, for another agent to pick up.
Status: **open, unclaimed.** Context: the muse-glimmer onboarding, #463; the fix this builds on, #464.

## What was seen

The vision suite's scene grounding (`scene_single`, six objects, norm-1000 boxes) and document serial, thinking on,
`num_ctx` 16384, greedy, Metal, rendered by `summarize_engine_compare.py` from the score files on the
`task/muse-glimmer-onboard` worktree:

| tag | build | Scene bbox IoU | Boxes / labels / colors | Serial |
|---|---|---|---|---|
| `30b-q8_0` | release `0.40.2-dynres-0-g5623192` | 0.000 | 0/0 · 0/6 · 0/6 | ❌ |
| `30b-q8_0-dflash` | release | 0.000 | 0/0 · 0/6 · 0/6 | ❌ |
| `30b-q8_0` | #464 `0.40.2-dynres-2-g4b7c3b7` | 0.537 | 5/6 · 6/6 · 6/6 | ✅ |
| `30b-q8_0-dflash` | #464 | **not run** | | |
| `30b` (Q4_K_M) | #464 | 0.670 | 5/6 · 6/6 · 6/6 | ✅ |
| `30b-nvfp4` | #464 | 0.724 | 6/6 · 6/6 · 6/6 | ✅ |
| `30b-mxfp8` | #464 | 0.705 | 5/6 · 6/6 · 6/6 | ✅ |

Thinking off, #464's first commit `0.40.2-dynres-1-g4cfad4a`: `30b-q8_0` and `30b-q8_0-dflash` both 0.712, 6/6, serial ✅.

- **The release rows are not a measurement of the model.** On the release build, two-pass pass two lacked glimmer's
  message header, so the model echoed the prompt as JSON (`{"user": "You are a …`). #464's second commit fixes that.
  Every release think-on cell of the `-dflash` tags still carries it.
- **So the open question has two parts:**
  1. `30b-q8_0-dflash` has no think-on cell on a fixed build. Run it.
  2. `30b-q8_0` with thinking on scores 0.537 on the fixed build, lowest of the glimmer tags (0.670–0.724), with one
     box missing. Is that the quantization, the thinking, or noise?

## What to do

1. Build #464's branch (`fix/glimmer-format-think-off`), or `main` if #464 has merged. Stage it on :11437 against an APFS
   clone of production's store (`cp -cR`). Use production's environment: `OLLAMA_FORMAT_TWO_PASS=1`,
   `OLLAMA_KV_CACHE_TYPE=f16`, `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0`. Run under the host's GPU gate (production idle 15
   min, powermode 2). Never touch :11435.
2. **Pin `NUM_BATCH=2048` on every GGUF cell.** llama-server's automatic batch moves GGUF greedy output (memory note:
   GGUF automatic batch confound); an unpinned A/B is not comparable.
3. Run the suite think-on on `30b-q8_0-dflash` and again on `30b-q8_0`, two repeats each. GGUF greedy with no drafter
   should repeat byte for byte. If `30b-q8_0` repeats at 0.537, the number is the model's.
4. Run `scene_single` alone on `30b-q8_0` with thinking off and on. Read the boxes and the thinking text to find which
   object is missed, and whether the thinking names it.
5. Compare with stock `v0.40.2` in single pass (`OLLAMA_FORMAT_TWO_PASS=0`). If stock gets the box, the two-pass
   flow is implicated. Note that stock's single pass is upstream's only mode.
6. Report per tag and build. Render tables with `summarize_engine_compare.py`, paste them verbatim, and give the
   power mode and build of every row.

## Done when

- `30b-q8_0-dflash` has a think-on cell on a fixed build.
- The 0.537 is either reproduced and attributed (quantization, thinking, two-pass) or shown to be noise.
- If it is a fork bug, a fix PR exists; if it is the model, this record says so with the numbers.
