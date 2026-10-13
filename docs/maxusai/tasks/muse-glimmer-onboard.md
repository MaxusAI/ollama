# TASK: onboard Meta's muse-glimmer on Metal (MLX and GGUF, with and without its DFlash drafter)

Opened by the Metal host on 2026-10-11. Status: **done** (2026-10-13). Phases 0-5 ran on all eight pulled tags. The
results are in "Results" below; what is still open is in "Not covered".

## Outcome

**Five fixes came out of this onboarding. All five are merged on main and are in the `v0.40.2-dynres.1` point tag:**
- **#464, glimmer's format path:**
  - With thinking off, a format waits for glimmer's message header instead of binding before it. This was the GGUF `{...}<|eot|>` reply, in single pass and two-pass alike.
  - With thinking on, two-pass pass two opens the header before binding the format. Without it, the model echoed the prompt as JSON.
- **#465:** MLX reads `draft_num_predict` per request: 0 turns drafting off and N caps the depth.
- **#466:** MLX admission prices glimmer's KV cache. The estimator now reads the list form of `sliding_window_pattern`; before, it priced weights only.
- **#468:** an explicit `image_max_tokens: 1120` is honoured, and an unset budget reaches the runner as 0 (ADR 0048).
- **#470:** JSON whitespace is bounded at llama.cpp's limit on both runners. Glimmer in single pass padded MLX objects up to the old bound.

**Also from this campaign:**
- #469, the vision suite's loop detector (ADR 0049): merged.
- #467, the measured preflight rows for both Metal stacks: merged 2026-10-13, after a re-run on `v0.40.2-dynres.1`.

**Follow-ups, opened as task records for other agents (unclaimed):**
- #471: why Q8_0 grounds worse with thinking on, and the `30b-q8_0-dflash` think-on cell that never ran on a fixed build.
- #472: why no tag reads the 7 px fine text, and whether a larger image budget could help.
- #473: why a glimmer suite cell is 1.6-3× slower than a gemma4:31b cell.

**Retest on main.** `muse-glimmer:30b-q4_K_M-dflash` was re-run with thinking off on main `601b8a65c`, which is
code-identical to `v0.40.2-dynres.1` outside `docs/`. It reads fine text 4/4/4/4/0, with 0 errors and a scene IoU of
0.726, the same as on #464's build. The table is under phase 3a.

## The model, as the registry has it

`muse-glimmer` is a 30B (27.9B) text+image model with thinking and tools, 128K context, Apache 2.0
(ollama.com/library/muse-glimmer). Read from the registry's manifests and config blobs, 2026-10-11:

| tag (as pulled) | runner | weights | drafter | size | note |
|---|---|---|---|---|---|
| `30b-nvfp4` | MLX | NVFP4 (modelopt mixed) | none | 17.3 GB | |
| `30b-mlx` | MLX | NVFP4 | DFlash, 58 `draft.*` tensors | 19.1 GB | **the same manifest as `30b-nvfp4-dflash`**, not an 8-bit build |
| `30b-mxfp8` | MLX | MXFP8 | none | 32.6 GB | the 8-bit MLX build |
| `30b-mxfp8-dflash` | MLX | MXFP8 | DFlash | 35.2 GB | |
| `30b` | llama.cpp | Q4_K_M | none | 18.2 GB | the same manifest as `30b-q4_K_M` and `latest` |
| `30b-q4_K_M-dflash` | llama.cpp | Q4_K_M | DFlash GGUF, 1.6 GB `image.draft` layer | 19.8 GB | |
| `30b-q8_0` | llama.cpp | Q8_0 | none | ~31 GB | |
| `30b-q8_0-dflash` | llama.cpp | Q8_0 | DFlash GGUF | ~33 GB | |

- **Each (plain, dflash) pair differs only by the drafter.** The plain and dflash tags of each MLX pair share all
  1437 base tensor digests, and each GGUF pair shares its one model blob. Their modelopt quantization JSON differs, but only
  `ollama create` reads it.
- **Packaged parameters:** temperature 1, top_k 64, top_p 0.95 on every tag. The MLX dflash tags add
  `draft_num_predict: 15`, the GGUF ones `draft_num_predict: 3`.
- **Drafter:** `MuseGlimmerAssistantModel`, 5 sliding-window layers, block size 16.
  - MLX target layers `[1, 13, 25, 37, 49]`, GGUF `[2, 14, 26, 38, 50]`.
  - Presumably the same layers with a different index base; not verified.
- **Vision:** patch 14, downsample 2, so stride 28.

## What the code said before the campaign (origin/main 5623192cc = the release build)

The fork carried upstream's glimmer support unchanged, apart from MLX image budgets (ADR 0021), the
draft-under-grammar knob and the ADR 0039 drafter scale identity.
- **Thinking:** `false`, `"low"` … `"max"`; omitted or `true` is `"high"` (`model/renderers/glimmer.go:66-77`).
- **Images:** an aspect-preserving 28-pixel grid, never upscaled, capped at 4096 tokens, on both runners. GGUF
  ignores the image-token options, and its context fit charges 768 per glimmer image (`llm/llama_server.go:1297-1340`).
- **The expected traps, both now fixed:** MLX read `image_max_tokens: 1120` as unset (#468), and ignored
  `draft_num_predict` (#465).

## Phases

Every phase ran under production's environment: TWO_PASS=1, KV f16, DRAFT_UNDER_GRAMMAR=0 and MAX_LOADED_MODELS=1.
- **Where:** on `127.0.0.1:11437`, against an APFS clone of production's store (`~/.ollama/models-mlx-glimmer`).
- **Gate:** every GPU request waited for production to be idle 15 minutes, with powermode 2. A request was cancelled and re-run if production woke.
- **Drivers:** the onboarding worktree's `.campaign/` (not in git): `env.sh`, `stage-gl.sh`, `gate.sh`/`leg.sh`,
  `drv/drive-gl.sh`, and `campaign-gl{,-fix,-stock}.sh`.

**Builds, and which cells ran on each.** The version stamp is read from every score file's `server_version`:

| build | stamp | what ran on it |
|---|---|---|
| release (production then) | `0.40.2-dynres-0-g5623192` | phases 0-2, 4 and 5; phase 3 in both modes on all 8 tags (prefix `rgl_`) |
| #464, first commit `4cfad4ab8` | `0.40.2-dynres-1-g4cfad4a` | phase 3 think-off, all 8 tags (`rglfix_`) |
| #464, second commit `4b7c3b7c9` | `0.40.2-dynres-2-g4b7c3b7` | phase 3 think-on, the 4 plain tags (`rglfix_`) |
| stock ollama 0.40.2 | `0.40.2` | smoke on all 8 tags; phases 3 and 4 on `30b-nvfp4` and `30b` (`rglstock_`, `ocrglstock_`) |
| main `601b8a65c` | `0.40.2-dynres-35-g601b8a6` | phase 3 think-off, `30b-q4_K_M-dflash` only (the retest above) |

| phase | what | how |
|---|---|---|
| 0 | no-GPU gates | the glimmer Go tests; `step-inventory.sh`: manifests byte-identical to the registry, blobs complete, drafters where expected |
| 1 | functional smoke, all 8 tags | `leg.sh smoke` → `glimmer_probe.py` → `runs/smoke.json` |
| 2 | preflight baseline rows | `leg.sh ladder` → `measure_ladder.py`, arch `muse-glimmer`, stride 28 |
| 3 | vision suite, both think modes | 27 cases + finetext per cell, cold start, greedy. Think off: `num_ctx` 16384, `num_predict` 2200. Think on: at `"high"`, temperature 0 with the packaged top_p/top_k, `num_ctx` ladder from 16384 |
| 4 | OCRBench v1 rows 0-199, think off | `/api/generate`, `num_ctx` 16384, `num_predict` 512, temperature 0 |
| 5 | what the drafter buys | `gl_tps.py`: ABBA × 5 per pair, a 512-token prose answer with thinking off, and a think-high reasoning run |

**The release think-on cells are not a measurement of the model.** In that build, two-pass pass two lacked glimmer's header, so format requests echoed the prompt or looped. The think-on
cells to read are #464's. The release think-off cells take the format + think-off path #464's first commit fixes.
They are the "before"; #464's think-off cells are the measurement.

## Phase 0, run 2026-10-11

- **Go tests:** the glimmer tests in `model/renderers`, `model/parsers`, `create`, `server`, `mlxrunner`,
  `mlxrunner/model` and `mlxrunner/model/glimmer` pass at `v0.40.2-dynres`.
- **The inventory:** all eight tags match the registry, their blobs are complete, and every `-dflash` tag carries its drafter.

## Results

All suite and OCRBench cells are **n=1**. A difference between two cells is a lead, not a result, unless it
repeats. Throughput columns are single runs, and a surprising number is a trigger for a targeted re-run, not a
conclusion. The score files are gitignored, so they are not in git. They live on the Metal host, in the onboarding worktree's
`docs/maxusai/vision-suite/` (`scores_rgl*_1_muse-glimmer_*`, `ft_*`, `ext_ocrgl*_*`). Every table below is
`summarize_engine_compare.py`, `summarize_head_to_head.py`, `summarize_extbench.py` or `gl_tps.py` output, pasted as
printed.

### Phases 1-2

**Phase 1:** every smoke check passed on every tag except one. On the release build, `format_schema` with thinking off
failed on the four GGUF tags, returning `{...}<|eot|>`.
- **What the smoke found:** the four issues #464, #465, #466 and #468 fix (see "Outcome").
- **One open question:** MLX does not enforce `num_ctx` on prompt length. This is upstream ollama#18125, and it is
  intentional upstream.
- **Stock 0.40.2 fails more.** Its smoke fails `format_schema` on all eight tags:
  - GGUF fails with thinking off, as the release build did.
  - MLX fails with thinking on: stock has single pass only, and the answer is `{` followed by whitespace. #470 bounds that whitespace.
- **#464's build passes `format_schema`:**
  - in single pass on `30b-nvfp4`, `30b-mxfp8` and `30b` (`runs/1pass-464.json`);
  - in two-pass on `30b` and `30b-q4_K_M-dflash` (`runs/smoke-fix.json`).

**Phase 2:** both runners measured the same ladder, `[47, 182, 779, 2995, 4082]`. The rows are in #467, merged.

### Phase 3a, thinking off

**The measurement: #464's first commit (`0.40.2-dynres-1-g4cfad4a`), all eight tags.** Rendered with
`summarize_engine_compare.py --dir . --think false --prefix rglfix_1_ <8 tags>`. This is the same output as `.campaign/runs/quality-off-glimmer.md`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| muse-glimmer:30b | GGUF | 16384 | 0.726 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-mlx | **MLX** | 16384 | **0.823** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | **0.823** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-mxfp8 | **MLX** | 16384 | **0.681** | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-mxfp8-dflash | **MLX** | 16384 | **0.681** | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-q4_K_M-dflash | GGUF | 16384 | 0.726 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-q8_0 | GGUF | 16384 | 0.712 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |
| muse-glimmer:30b-q8_0-dflash | GGUF | 16384 | 0.712 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muse-glimmer:30b | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 756 | 12 | 139 | 85.8 | 42 |
| muse-glimmer:30b-mlx | **MLX** | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 735 | 20 | 3229 | 37.0 | 97 |
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 735 | 22 | 3541 | 34.4 | 105 |
| muse-glimmer:30b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1127 | 11 | 4663 | 107.5 | 33 |
| muse-glimmer:30b-mxfp8-dflash | **MLX** | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1127 | 11 | 2622 | 106.9 | 34 |
| muse-glimmer:30b-q4_K_M-dflash | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 756 | 26 | 214 | 44.1 | 82 |
| muse-glimmer:30b-q8_0 | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1367 | 11 | 198 | 146.4 | 25 |
| muse-glimmer:30b-q8_0-dflash | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1367 | 22 | 238 | 76.2 | 47 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2-dynres-1-g4cfad4a · think=false

- **Document extraction is perfect on every tag.** Every tag gets 5/5 items, 5/5 quantity and price, the total and the serial.
- **Scene boxes** are glimmer's weakness against gemma4 (below). nvfp4 scores 0.823 and the 8-bit MLX build 0.681. With n=1 and no bf16 arm, that
  gap cannot be put on quantization.
- **Fine text** is 4/4 at 22, 16 and 12 px and 3-4/4 at 9 px, and **0/4 at 7 px on every tag** (#472).
- **The drafter changed no score.** Each (plain, dflash) pair has identical quality columns and the same answer token count.
  Each pair was one run per tag, greedy.

**Against gemma4:31b, thinking off.** gemma4's cells are the v0.40.2 fold's gate 6 (`r0402fold_`, a different build
and port). The generator marks the table MIXED for that reason. Rendered with `summarize_head_to_head.py --tags`, from one directory
holding both sets of score files:

| test | metric | rglfix_1_muse-glimmer_30b-nvfp4_thinkfalse | r0402fold_1_gemma4_31b-nvfp4_thinkfalse | rglfix_1_muse-glimmer_30b_thinkfalse | r0402fold_1_gemma4_31b-it-q4_K_M_thinkfalse |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.823 (16384) | 0.962 (16384) | 0.726 (16384) | 0.964 (16384) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.626 (16384) | 0.750 (16384) | 0.697 (16384) | 0.751 (16384) |
| fine text | 22/16/12/9/7 px | 4/4/4/3/0 (16384) | 4/4/4/3/3 (16384) | 4/4/4/4/0 (16384) | 4/4/4/4/3 (16384) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| multi (3 img, anchored) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| throughput | gen tok/s | 22 | 19 | 12 | 12 |
| throughput | prefill tok/s | 3541 | 1649 | 139 | 2089 |
| latency | s/req (unique image) | 34.4 | 30.0 | 85.8 | 45.5 |
| latency | req/h (serial) | 105 | 120 | 42 | 79 |

Provenance (from score files): host(s) http://127.0.0.1:11436, http://127.0.0.1:11437 · build(s) 0.35.0-dynres-27-g948ef3a, 0.40.2-dynres-1-g4cfad4a ⚠ MIXED — columns are not one campaign

- **Document fields:** a tie.
- **gemma4 is ahead on:** scene boxes (0.82 MLX / 0.73 GGUF against 0.96), name boxes, and 7 px text (0/4 against 3/4).
- **The interim record's speed lead does not hold.** It said glimmer was faster, but that came from release cells that stopped early on the format bug. On #464's build, glimmer
  takes 34 s against 30 s per request on MLX and 86 s against 46 s on GGUF (n=1; #473).

**Retest on main `601b8a65c` (`0.40.2-dynres-35-g601b8a6`), `30b-q4_K_M-dflash` only.** The bundle is committed on
branch `bench/0402-metal`, at
`docs/maxusai/vision-suite/bench-runs/vision-campaign-2026-10-13-metal-0402-glimmer-main.json`. Rendered with
`summarize_engine_compare.py --bundle <that file> --think false --prefix m0402fix_1_ muse-glimmer:30b-q4_K_M-dflash`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| muse-glimmer:30b-q4_K_M-dflash | GGUF | 8192/16384 ⚠ | 0.726 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muse-glimmer:30b-q4_K_M-dflash | GGUF | 8192/16384 ⚠ | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 756 | 34 | 282 | 33.8 | 106 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2-dynres-35-g601b8a6 · think=false

- **The quality columns match #464's cell for this tag.** It ran at that campaign's GGUF settings, and the generator flags its
  `num_ctx` (8192/16384) as mixed, so its speed columns do not compare with the cells above.

**The "before": the release build (`0.40.2-dynres-0-g5623192`), all eight tags. These are not glimmer's score.** They
take the format + think-off path #464 fixes, so they include:
- fine text 0/0/0/0/0 on every GGUF and MXFP8 tag;
- capped cells, with `multi_3img` on nvfp4 climbing to 131072;
- three GGUF HTTP 500s ("token repeat limit"): two on `30b`, one on `30b-q4_K_M-dflash`.

#464 has the before and after. To render this table, run
`summarize_engine_compare.py --dir . --think false --prefix rgl_1_ <8 tags>` on the Metal host.

### Phase 3b, thinking on

**The measurement: #464's second commit (`0.40.2-dynres-2-g4b7c3b7`), the four plain tags.** Rendered with
`summarize_engine_compare.py --dir . --think on --prefix rglfix_1_ <4 plain tags>`. This is the same output as `.campaign/runs/quality-on-glimmer-fix.md`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| muse-glimmer:30b | GGUF | 16384 | 0.670 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | **0.724** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-mxfp8 | **MLX** | 16384 | **0.705** | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-q8_0 | GGUF | 16384 | 0.537 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muse-glimmer:30b | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1702 | 18 | 251 | 106.8 | 34 |
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1590 | 22 | 942 | 75.8 | 48 |
| muse-glimmer:30b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1432 | 12 | 886 | 118.5 | 30 |
| muse-glimmer:30b-q8_0 | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2681 | 14 | 275 | 201.0 | 18 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2-dynres-2-g4b7c3b7 · think=on

- **All 27 cases finished at `num_ctx` 16384 on every tag,** with no loop and no cap.
- **Thinking does not help glimmer here.** Scene IoU is lower than thinking off on three of four tags. Q8_0's
  0.537 is the lowest (#471).
- **Fine text** is unchanged from thinking off, including 0/4 at 7 px.

**gemma4:31b-nvfp4, thinking on, for reference.** This is the regression A/B's release cell (`rrg_`, same build and port). Rendered with
`summarize_engine_compare.py --dir . --think on --prefix rrg_1_ gemma4:31b-nvfp4`. This is the same output as `.campaign/runs/quality-on-gemma.md`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.966** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 2961 | 26 | 212 | 122.3 | 29 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2-dynres-0-g5623192 · think=on

**The release build, all eight tags. This is not a measurement of the model.** Two-pass pass two lacked glimmer's header. The
results:
- `30b-mxfp8` echoed the prompt as `{"user": "You are a ..."}`.
- The loop detector (ADR 0049) ended 2-3 cases per cell on `30b-mxfp8(-dflash)`, `30b-q4_K_M-dflash` and `30b-q8_0(-dflash)`, and 1 on `30b`.
- The scene zeros on `30b`, `30b-q8_0` and `30b-q8_0-dflash` are on the same path: valid JSON, but no labelled objects in it.

**These are the only think-on cells for the four `-dflash` tags.** None was re-run on a fixed build. Rendered with
`summarize_engine_compare.py --dir . --think on --prefix rgl_1_ <8 tags>`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| muse-glimmer:30b | GGUF | 16384 | 0.000 | 0/0 · 0/6 · 0/6 | ✅ | 0/5 · 0/5 · ❌ | 0/5 |
| muse-glimmer:30b-mlx | **MLX** | 16384 | **0.733** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | **0.733** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |
| muse-glimmer:30b-mxfp8 | **MLX** | 16384 | **0.724** | 6/6 · 6/6 · 6/6 | ✅ | 0/5 · 0/5 · ❌ | 0/5 |
| muse-glimmer:30b-mxfp8-dflash | **MLX** | 16384 | **0.743** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-q4_K_M-dflash | GGUF | 16384 | 0.657 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-q8_0 | GGUF | 16384 | 0.000 | 0/0 · 0/6 · 0/6 | ❌ | 5/5 · 5/5 · ✅ | 4/5 |
| muse-glimmer:30b-q8_0-dflash | GGUF | 16384 | 0.000 | 0/0 · 0/6 · 0/6 | ❌ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muse-glimmer:30b | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ❌ q1_right, q2_right, q4_bbox_hit | — | 2464 | 16 | 221 | 170.4 | 21 |
| muse-glimmer:30b-mlx | **MLX** | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1336 | 27 | 953 | 52.8 | 68 |
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1398 | 20 | 888 | 72.1 | 50 |
| muse-glimmer:30b-mxfp8 | **MLX** | 16384 | 4 | 4 | 4 | 4 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1137 | 12 | 998 | 101.8 | 35 |
| muse-glimmer:30b-mxfp8-dflash | **MLX** | 16384 | 4 | 4 | 4 | 4 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 2012 | 19 | 686 | 112.6 | 32 |
| muse-glimmer:30b-q4_K_M-dflash | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ❌ q1_right, q2_right, q4_bbox_hit | — | 1784 | 25 | 228 | 85.4 | 42 |
| muse-glimmer:30b-q8_0 | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ❌ q1_right, q2_right, q4_bbox_hit | — | 2409 | 12 | 301 | 218.3 | 16 |
| muse-glimmer:30b-q8_0-dflash | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ❌ q1_right, q2_right, q4_bbox_hit | — | 2429 | 23 | 225 | 120.8 | 30 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2-dynres-0-g5623192 · think=on

### Phase 3, stock ollama 0.40.2

**The upstream comparison, `30b-nvfp4` and `30b`.** Stock has no two-pass, so think-on is single pass.

**Thinking off** fails the same way the release build did, which makes it upstream's bug (ollama#18687 covers its single-pass
half):
- nvfp4's document comes back 0/5;
- `multi_3img` is capped at 131072;
- the GGUF fine text is 0/0/0/0/0, with two HTTP 500s.

Rendered with `summarize_engine_compare.py --dir . --think false --prefix rglstock_1_ muse-glimmer:30b-nvfp4 muse-glimmer:30b`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| muse-glimmer:30b-nvfp4 | **MLX** | 16384/131072 ⚠ | **0.839** | 6/6 · 6/6 · 6/6 | ✅ | 0/5 · 0/5 · ❌ | 0/5 |
| muse-glimmer:30b | GGUF | 16384 | 0.711 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muse-glimmer:30b-nvfp4 | **MLX** | 16384/131072 ⚠ | 4 | 4 | 4 | 4 | 0 | capped | ✅ q1 + q2 + q4-bbox | — | 357 | 13 | 4474 | 28.8 | 125 |
| muse-glimmer:30b | GGUF | 16384 | 0 | 0 | 0 | 0 | 0 | ❌ q1_right, q2_right, q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 357 | 25 | 5364 | 15.1 | 238 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2 · think=false

**Thinking on** is clean, and close to #464's build: scene 0.732 against 0.724 on MLX, and 0.670 against 0.670 on GGUF (n=1).
Rendered with `summarize_engine_compare.py --dir . --think on --prefix rglstock_1_ muse-glimmer:30b-nvfp4 muse-glimmer:30b`:

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | **0.732** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3/5 |
| muse-glimmer:30b | GGUF | 16384 | 0.670 | 5/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| muse-glimmer:30b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1322 | 21 | 3522 | 64.4 | 56 |
| muse-glimmer:30b | GGUF | 16384 | 4 | 4 | 4 | 4 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 1701 | 12 | 160 | 159.3 | 23 |

Provenance (from score files): host(s) http://127.0.0.1:11437 · build(s) 0.40.2 · think=on

### Phase 4, OCRBench v1 rows 0-199, thinking off

**The release build, all eight tags.** OCRBench sends no format, so it does not take #464's path and was not re-run. Rendered with
`summarize_extbench.py ocrbench ocrgl_muse-glimmer_<tag> ...`:

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `muse-glimmer:30b-nvfp4` | 200 | 0 | 0 | 179 | **0.895** | false | generate |
| `muse-glimmer:30b-mlx` | 200 | 0 | 0 | 180 | **0.9** | false | generate |
| `muse-glimmer:30b-mxfp8` | 200 | 0 | 0 | 183 | **0.915** | false | generate |
| `muse-glimmer:30b-mxfp8-dflash` | 200 | 0 | 0 | 183 | **0.915** | false | generate |
| `muse-glimmer:30b` | 200 | 0 | 0 | 182 | **0.91** | false | generate |
| `muse-glimmer:30b-q4_K_M-dflash` | 200 | 0 | 0 | 182 | **0.91** | false | generate |
| `muse-glimmer:30b-q8_0` | 200 | 0 | 0 | 182 | **0.91** | false | generate |
| `muse-glimmer:30b-q8_0-dflash` | 200 | 0 | 0 | 182 | **0.91** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

host: http://127.0.0.1:11437 · build: 0.40.2-dynres-0-g5623192 · prompt_cache_ram: 0

**Stock 0.40.2.** Rendered with `summarize_extbench.py ocrbench ocrglstock_muse-glimmer_30b-nvfp4 ocrglstock_muse-glimmer_30b`:

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `muse-glimmer:30b-nvfp4` | 200 | 0 | 0 | 179 | **0.895** | false | generate |
| `muse-glimmer:30b` | 200 | 0 | 0 | 182 | **0.91** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

host: http://127.0.0.1:11437 · build: 0.40.2 · prompt_cache_ram: 0

- **Every tag scores 0.895-0.915.** On one 200-item slice the standard error is about 2 points, so the tags are not told apart.
- **The drafter and the fork change almost no items.** `summarize_extbench.py --paired`, run pair by pair, finds:
  - `30b-nvfp4` against `30b-mlx`: 1 discordant item;
  - each of the other three (plain, dflash) pairs: 0;
  - release against stock, on `30b-nvfp4` and on `30b`: 0.

### Phase 5, what the drafter buys

**The release build, 2026-10-12.** The run was 5 reps per arm in ABBA order, temperature 0:
- prose: thinking off, 512 tokens;
- reason: think `"high"`, the bat-and-ball question, run to completion.

The median decode tok/s per (plain, dflash) pair, as `gl_tps.py` prints it. It was re-rendered from `runs/tps.json` with `--reps 0`, which sends no requests:

```
== median decode tok/s
prose  muse-glimmer:30b-nvfp4         17.88   muse-glimmer:30b-mlx               22.75   x1.27
reason muse-glimmer:30b-nvfp4         18.27   muse-glimmer:30b-mlx               28.64   x1.57
prose  muse-glimmer:30b-mxfp8         10.25   muse-glimmer:30b-mxfp8-dflash      14.40   x1.40
reason muse-glimmer:30b-mxfp8         10.53   muse-glimmer:30b-mxfp8-dflash      16.24   x1.54
prose  muse-glimmer:30b               15.93   muse-glimmer:30b-q4_K_M-dflash     15.61   x0.98
reason muse-glimmer:30b               15.57   muse-glimmer:30b-q4_K_M-dflash     22.52   x1.45
prose  muse-glimmer:30b-q8_0          10.69   muse-glimmer:30b-q8_0-dflash       13.26   x1.24
reason muse-glimmer:30b-q8_0          10.89   muse-glimmer:30b-q8_0-dflash       18.82   x1.73
```

- **Drafting speeds decode 1.24-1.73×** on every pair and workload but one. The exception is GGUF Q4_K_M prose, at 0.98×.
- **Acceptance per request** is in `runs/tps.json`. MLX reports it as `speculative decode stats`, llama-server as `draft acceptance`.
- **These are 5 reps of one prompt each,** run on the release build and before #465. Treat them as a lead for #473, not a ranking of the tags.

## Changes in this branch

- `docs/maxusai/vision-suite/sampling.py`: a `muse-glimmer` think-on entry.
  - Temperature comes from `THINK_TEMPERATURE` (0).
  - top_p 0.95 and top_k 64 come from the packaged params, the only published values.
  - Without it, think-on cells would send no sampling keys and run at the packaged temperature 1.
- **This record.** The measured `[expect.*.muse-glimmer]` rows and the `metal-0-40-2` profile are in #467. The loop
  detector landed through #469.

## Not covered

- **Think-on on a fixed build for the four `-dflash` tags.** Their only think-on cells are the release ones above, which are not a
  measurement. #471 asks for `30b-q8_0-dflash`; `30b-mlx`, `30b-mxfp8-dflash` and `30b-q4_K_M-dflash` are not
  claimed by anyone.
- **OCRBench with thinking on,** and OCRBench on #464's build. The OCRBench path sends no format, so #464 should not change it.
- **The bf16 tags** were not pulled. Without a bf16 arm, a quality gap between NVFP4, MXFP8, Q4 and Q8 cannot be put on
  quantization (the tier-move rule).
- **Repeats.** Every suite and OCRBench cell is n=1.
- **Not in the suite:** think levels other than `"high"` (the phase-1 probe covers them), audio, and tools beyond one round trip.
- **CUDA and ROCm.** This was the Metal host's run.

`macbook-pro-m5-max-128GB/mlx-metal`
