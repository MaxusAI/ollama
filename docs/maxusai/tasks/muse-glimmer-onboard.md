# TASK: onboard Meta's muse-glimmer on Metal (MLX and GGUF, with and without its DFlash drafter)

Opened by the Metal host on 2026-10-11, while the maintainer pulls the tags. Status: **prepared, not run.** Nothing below
has touched a GPU yet; the no-GPU gates have run (section "Phase 0").

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

- **Each (plain, dflash) pair differs only by the drafter.** All 1437 MLX base tensor digests are identical between
  `30b-nvfp4` and `30b-nvfp4-dflash`, and between `30b-mxfp8` and `30b-mxfp8-dflash`; each GGUF pair shares its one
  model blob. Their quantization metadata JSON differs (different modelopt producer versions; the plain NVFP4 build
  lists 300 `vision_encoder` layers the dflash build does not), but that JSON is read only by `ollama create`
  (`create/create.go`), not by the runner. The no-draft control in phase 1 checks the claim.
- **Packaged parameters:** temperature 1, top_k 64, top_p 0.95 on every tag. The MLX dflash tags add
  `draft_num_predict: 15`, the GGUF ones `draft_num_predict: 3`.
- **Drafter:** `MuseGlimmerAssistantModel`, 5 sliding-window layers, block size 16. It reads target layers
  `[1, 13, 25, 37, 49]` (MLX `dflash_config.target_layer_ids`); the GGUF drafter's `dflash.target_layers` is
  `[2, 14, 26, 38, 50]`. That is presumably the same layers with a different index base; not verified.
- **Vision:** patch 14, downsample 2, so stride 28.
- The 8 pulled tags hold 102.8 GB of unique blobs. 603 GB was free when the pulls started.

## What the code says (origin/main 5623192cc = production's 0.40.2-dynres-0-g5623192)

The fork carries upstream's glimmer support unchanged, except for MLX image budgets (ADR 0021), the
draft-under-grammar knob, and the ADR 0039 scale identity in the DFlash drafter.

- **Thinking:** think accepts `false`, `"low"`, `"medium"`, `"high"` and `"max"`. Omitted or `true` means `"high"`
  (`model/renderers/glimmer.go:66-77`). Any other value is a 400 (`api/types.go:1255`). `false` renders
  "Reasoning strength: none." and the parser drops any thinking that is produced anyway.
- **Images, both runners:** aspect-preserving grid, never upscaled, capped at 4096 tokens.
  - MLX: `mlxrunner/model/glimmer/media.go:21,125`.
  - llama.cpp b11351: `PROJECTOR_TYPE_MUSE_GLIMMER`, `set_limit_image_tokens(1, 4096)`.
  - None of the fork's clip patches touch this projector.
- **Request image-token options:**
  - MLX honours `image_max_tokens` through `ResolveImageBudget`. Expected trap: a value of 1120
    (`api.DefaultImageMaxTokens`) reads as unset, so 4096.
  - GGUF ignores them: `muse-glimmer` is not in `llm/llama_server.go` `visionServerArgs`.
- **GGUF context fit:**
  - `MaxImageTokens` and `ImageTokensForSize` have no glimmer case, so the fit check charges 768 per image
    (`llm/llama_server.go:1297-1340`), against a real cost of up to about 4098.
  - Many large images against a small `num_ctx` is the case to probe.
- **Drafting:**
  - MLX drafts whenever the manifest has a drafter, at an adaptive depth. It does not draft for logprobs requests
    (`mlxrunner/speculate.go:148`) or, in production's environment, under a format/grammar (`DRAFT_UNDER_GRAMMAR=0`).
    It does not read `draft_num_predict`.
  - GGUF passes `--spec-type draft-dflash --spec-draft-n-max <draft_num_predict> --spec-draft-model <blob>`
    (`llm/llama_server.go:862-890`). llama-server logs `draft acceptance = …`.

## Phases

Everything runs on production's own release build, `~/.ollama/binaries/stage-0.40.2-dynres-0-g5623192`, which is
pinned by `EXPECT_STAMP`. It runs:
- on `127.0.0.1:11437`;
- in production's environment: TWO_PASS=1, KV f16, DRAFT_UNDER_GRAMMAR=0, and MAX_LOADED_MODELS=1;
- against an APFS clone of production's store, `~/.ollama/models-mlx-glimmer`.

Every GPU request waits until production has been idle 15 minutes with powermode 2, and is cancelled and re-run if
production wakes. The drivers live in the worktree's `.campaign/`:

- `env.sh`: the build, store, port, tags and pairs.
- `stage-gl.sh`: the stage, which is also the restart hook.
- `gate.sh` and `leg.sh`: the per-step gate.
- `drv/drive-gl.sh`: the per-request gate, from `drive-0402.sh` with its port and paths repointed.

| phase | what | how | verdict |
|---|---|---|---|
| 0 | no-GPU gates | the glimmer Go tests (below); `step-inventory.sh`: manifests byte-identical to the registry, blobs complete, drafters where expected; clone the store | pass/fail |
| 1 | functional smoke, all 8 tags | `leg.sh smoke` → `glimmer_probe.py` → `runs/smoke.json` | PASS/FAIL + INFO |
| 2 | preflight baseline rows | `leg.sh ladder` → `measure_ladder.py`, arch `muse-glimmer`, stride 28: `30b-nvfp4` on `mlx-metal-0-40-2`, `30b` on a new `metal-0-40-2` | rows to paste, measured |
| 3 | vision suite, both think modes | `campaign-gl.sh` (3a off, 3b on at `"high"`): 27 cases + finetext per cell, every tag | scores (ADR 0012 tables) |
| 4 | OCRBench v1 rows 0-199, think off | `campaign-gl.sh` | accuracy per tag |
| 5 | what the drafter buys | `leg.sh tps` → `gl_tps.py`: ABBA × 5 per pair, prose think-off and a think-high reasoning run | tok/s, acceptance |

### Phase 1, the probe's checks (`glimmer_probe.py`)

Each check below is PASS/FAIL against a known right answer:
- `show`: the capabilities include completion, vision, tools and thinking.
- `text_think_off`: 17 × 23 = 391, with no thinking.
- `think_levels`: false, low, medium, high, max and omitted, on the bat-and-ball question (0.05). There must be
  thinking exactly when it is not `false`. It records tokens to finish at each level, uncapped (num_predict 24576).
- `think_invalid_level`: "extreme" must return 400.
- `tools_round_trip`: think off and default. It must call `get_weather(city=Paris)` and then use the tool's 18 °C.
- `format_schema`: think off and default. The output must be valid JSON for the schema, with thinking exactly when
  on (TWO_PASS=1).
- `image_read`: the invoice number `INV-2026-0801` off `visimgs/document.png`.

Each check below is INFO, measured with no expectation yet:
- `route`: which runner served the tag, the llama-server flags, and whether a drafter loaded.
- `image_token_ladder`: image tokens for 448², 768², 1024×768, 1920×480, 1568², 2048² and 2560×1440 test cards.
  Each count is the difference against the same text-only prompt, with a fresh system nonce per request so the
  prompt cache cannot shrink it.
- `image_token_options`: `image_max_tokens` set to unset, 1024, 1119, 1120, 1121 and 2048 on a 2048² card, plus
  `image_min_tokens` 2048 on a 448² card. This is where the 1120 trap and the GGUF no-op show.
- `ctx_fit_3x2048_at_8k`: three 2048² cards at num_ctx 8192. An error is the honest answer; a 200 with a short
  `prompt_eval_count` is silent truncation.
- `longgen`: 600 greedy tokens, run twice, plain and with `logprobs`, which is MLX's no-draft control. It records
  tok/s, the drafting lines from the log, and whether the text equals the plain tag's.

### What would count as a finding

- **Any phase-1 FAIL.**
- **Drafting changes greedy text.** A drafting tag's greedy text differs from its plain tag's and, on MLX, the
  logprobs control matches. That is the drafter itself changing greedy output. Verification is not exact on Metal (drafted text depends on draft depth, which is
  wall-clock driven; the `drv/drive-gl.sh` header).
- **The logprobs control differs too.** That means the two tags do not load identically despite identical tensors.
- **GGUF silently truncates images** in `ctx_fit_3x2048_at_8k`.
- **MLX and GGUF image-token counts disagree** at equal sizes. They are meant to run one algorithm.

## Phase 0, run 2026-10-11

On the release worktree at `v0.40.2-dynres`, which is production's build:

```
ok  	github.com/ollama/ollama/model/renderers	0.746s
ok  	github.com/ollama/ollama/model/parsers	1.255s
ok  	github.com/ollama/ollama/create	0.530s
ok  	github.com/ollama/ollama/server	0.918s
ok  	github.com/ollama/ollama/mlxrunner	0.690s
ok  	github.com/ollama/ollama/mlxrunner/model	0.471s
ok  	github.com/ollama/ollama/mlxrunner/model/glimmer	0.508s
```

The commands were:
- `go test ./model/renderers ./model/parsers ./create -run Glimmer`
- `./server -run TestGlimmer`
- `./mlxrunner -run 'TestDraftUnderGrammarFromEnv|TestDraftingEnabled|TestDFlash|TestDecodeBlockDraft'`
- `./mlxrunner/model -run TestResolveImageBudget`
- `./mlxrunner/model/glimmer -run 'TestParseConfig|TestParseOfficialHFConfig|TestValidateTokenizerEOS|TestComputeImageSize|TestPrepareMedia'`

`step-inventory.sh` runs once the pulls finish.

## Changes in this branch

- `docs/maxusai/vision-suite/sampling.py`: a `muse-glimmer` think-on entry. Temperature comes from
  `THINK_TEMPERATURE` (0), and top_p 0.95 / top_k 64 from the packaged params, which are the only published values.
  Without it, think-on cells would send no sampling keys and run at the packaged temperature 1.
- **After phase 2:** the measured `[expect.*.muse-glimmer]` rows. `muse-glimmer` joins `mlx-metal-0-40-2`'s arches,
  and a `metal-0-40-2` profile is added with its version-pattern test. The newest llama.cpp-path Metal profile is
  `metal-0-32-14`, so 0.40.2's GGUF path has no profile today.

## Not covered

- **The bf16 tags** (`30b-bf16(-dflash)`, `30b-mlx-bf16(-dflash)`) were not pulled. With no bf16 arm, a quality gap
  between NVFP4/MXFP8/Q4/Q8 cannot be attributed to quantization (the tier-move rule).
- **Think levels in the suite.** The suite measures think-on at the default `"high"` only; levels appear only in the
  phase-1 probe.
- **Audio** (glimmer declares none) and **tools beyond one round trip.**
- **CUDA and ROCm.** This is the Metal host's run.

`macbook-pro-m5-max-128GB/mlx-metal`
