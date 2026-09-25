# TASK: fold upstream v0.34.2 into main

Branch `task/upstream-sync-0.34.2`, worktree `claude-scratch/wt-sync0342`.

## Status (2026-09-20, deployed)

| gate | state |
|---|---|
| 1, the merge | **done** — `go build` clean, `go vet` clean, `go test` 57 packages ok |
| 2, docs and paths | **done** — 53 files re-pointed, link check clean |
| 3, the patch series against b10969 | **done** — 6 of 7 apply clean; 906 retired as obsolete |
| 4, image build | **done** — `maxusai/ollama:sync-0.34.2`, 2 h 16 m, rc=0 |
| 5, preflight | **PASS 21 / SKIP 4** on a canary, after the payload pin moved with evidence |
| 6, campaigns | **done** — every scored cell equal to the deployed build, GGUF and MLX |
| tag and deploy | **done** — `v0.34.2-dynres`, `:11497` running `0.34.2-dynres-0-g5bffaac` since 13:08:53 |
| post-deploy, on the shipped image | **done** — GGUF 8,172 cells unmoved, MLX within its own spread, OCRBench flat on four arms |

## What v0.34.2 changes for the fork

Fifteen commits. One is structural and the rest are small.

**The MLX engine leaves `x/`.** `x/mlxrunner` → `mlxrunner`, `x/mlxrunner/mlx` → `mlx`,
`x/create` → `create`, `x/models` → `mlxrunner/model`, `x/tokenizer` →
`mlxrunner/tokenizer`, `x/internal/mlxtest` → `mlx/mlxtest`, `x/internal/mlxthread*` →
`mlx/mlxthread*`. `nn` is laid out one layer kind per file and `model` by contract,
checkpoint and construction. Manifests move to a top-level `manifest` package, which
replaces `x/imagegen/manifest`. Three small packages fold into their users and the runner
launches without the engine dispatcher, so `--mlx-engine` is gone from the spawn.

**llama.cpp moves b10864 → b10969.** Upstream's own bump commit notes the build now
produces duplicate symbols between `libllama` and `libmtmd` and moves their compat shim
into `libllama` with exported symbols — which is the reason gate 3 exists.

**MLX and MLX-C do not move.** Both pins are identical to ours (`d9add9d1`, `ebc88f10`),
so the vision goldens, the #312 encoder work and the kernel questions are untouched by
this fold. That is what makes a 395-file diff a relocation rather than a risk.

**`ec3cc2307` is already ours** — the speculative-decode KV release we cherry-picked into
the 0.34.1 fold is one of the fifteen.

## How to merge a fold that moves the tree

Three approaches were tried and the two that look helpful are the worst:

| approach | conflicts | why |
|---|---|---|
| **merge as-is** | **51** | git's rename detection carries our *modified* files to the new paths and merges most of them |
| relocate our tree first | 121 | our move plus upstream's move is rename/rename; every auto-merge becomes manual |
| rewrite our imports first | 136 | pre-editing a file upstream also moved turns a clean rename into a content conflict |

So: **merge as-is, and fix only what git gets wrong.** What it gets wrong is our
*fork-only* files. Directory-rename inference picks the dominant target, and because more
files went to `mlx/` than to `mlxrunner/`, git suggests the repository **root** for
`x/mlxrunner` files — 14 of ours land there. Place those from upstream's own rename map
(`git diff --name-status -M v0.34.1..v0.34.2`), not from the suggestion.

## What was resolved by hand

- **11 content conflicts.** Nine were the `base` package becoming `model` plus moved
  import paths, taken as ours-plus-rewrite. Two carried real upstream changes:
  `client.go` (the manifest API — `ParseNamedManifest` over `LoadManifest`, summing
  `TensorLayers()` over `TotalTensorSize()` — and the spawn losing `--mlx-engine`) and the
  CI path filter, which takes upstream's new native paths plus the fork's own
  `llama/llama.cpp`, `ml/backend/ggml` and `.github` entries.
- **40 placements** of fork-only files: `kvsize`, `bench/qqmm`, the `client_*_test.go`
  set, the vision goldens and e2e tests, `tokenizer_special_test.go`.
- **gemma4 audio stays deleted.** Upstream modified `process_audio.go` and its test; the
  fork does not ship gemma4 audio (ADR 0021), so the deletion is kept.
- **`Root.Close` is gone upstream**, where it was a no-op. The two fork tests that called
  it rely on the open alone now.
- Two duplicate imports created by the path rewrite, and one file left declaring
  `package base`, were caught by the build rather than by review — worth remembering that
  the compiler is the cheap check here.

## Fork features verified present after the move

`kvsize`, `bench/qqmm`, `xgrammar`, the vision goldens, `llama/compat`, the
`MediaBudgetModel` seam (ADR 0021), the image-token budget options (ADR 0008), the
drafting knob (ADR 0033), the gemma4 batch floor (ADR 0036) and `ToMLXGlobalScale`
(ADR 0039, proposed).

## Documentation sweep

53 files under `docs/` and `.claude/` named a path that moved. Every such reference was
re-pointed, including inside ADRs and task docs: a path is a pointer, not a decision, so
rewriting it keeps history readable without changing what was decided. References to
things that were **deleted** rather than moved — `x/structured`, `x/imagegen` — are left
as written, because there is nowhere to point them. The link checker reports zero broken
relative links introduced by this fold; the 43 in `docs/design/gemma4-vision-token-budgets.md`
use an `ollama/`-prefixed convention and were already broken on main.

## Gate 3: the patch series against b10969

Dry-run with plain `git apply --check` on a b10969 checkout before building anything, per
the lesson the 0.34.1 fold learned the expensive way.

| patch | result |
|---|---|
| 001 hooks, 002 nemotron-dynres, 004 gemma4 budget-fill, 005 dynres pinned overshoot | apply clean |
| 801 clip node-stats meter, 903 MMQ ids padding | apply clean |
| **906 revert HIP integrated flag** | **does not apply — and should not** |

906 carried upstream's own revert `d4389a4dd92`, which our b10864 payload missed by 78
minutes. b10969 ships it: the source already reads `info.devices[id].integrated = false`
and no HIP-conditional assignment remains, so the patch is redundant and is retired here —
which is exactly what its own header instructed ("Drop this patch when the payload advances
past d4389a4dd92"). The defect it guarded, wrong output past `n_ubatch` on gfx1151, stays
fixed by upstream's code rather than by ours.

## Gates 4 and 5: the image and the canary (2026-09-19)

**Build**: `maxusai/ollama:sync-0.34.2`, 19:56 → 22:12 (2 h 16 m), rc=0, 5.43 GB, stamp
`0.34.1-dynres-26-g3dade56`, llama.cpp b10969, MLX `d9add9d1`, MLX-C `ebc88f10`, 222 payload
files — the same count as the deployed image. Twelve stages hit cache; the MLX stage did
**not**, despite an unchanged pin, because the stage's cache key covers the source tree and
this fold moved 300 files. That cost is specific to this fold.

**Preflight on a canary** (`:11530`, GPU0, 16 GiB reserve, never `:11497`):

| run | result |
|---|---|
| first, pin still naming b10864 | **FAIL 1 / PASS 20 / SKIP 4** — `payload_pin` only |
| after moving the pin with evidence | **PASS 21 / SKIP 4**, `VERDICT: PASS` |

The failure was the harness working: it pins the payload every ladder below it was measured
on and refuses to trust them when it moves. What makes the update honest is the order — the
pin failed while everything it gates still ran and passed on b10969:

- token ladders 5/5 within ±2 on **all three** arches (nemotron_h_omni, gemma4, qwen35)
- payload proofs on all three: the fork's budget flags still reach llama.cpp
- pinned budgets: gemma4 560 → 529, nemotron 3328 → 3270
- text baselines 19 / 19 / 13, unchanged
- the qwen2.5vl fp16-accumulate poison probe decodes healthily

So the rows were verified against the new payload *before* the pin line was edited, which
is the order `payload_pin`'s own message demands, and they are unchanged because b10969 did
not change them. The four skips are correct: three arches have no aspect-ladder expectation
recorded, and qwen35's pinned budget is arch-gated away.

Runs: `preflight-runs/full-0342-canary.{log,json}`, `canary-0342{,-rerun}.log`.

## Gate 6: the campaigns (2026-09-20)

Think-off throughout, `CTX_START=8192 CTX_MAX=65536`, one runner at a time, GPU0 with the
16 GiB reserve, never `:11497`. 13 suites in the two legs, **0 errors, 0 OOMs, 0 not
converged**, plus 8 control suites and 6 repeats.

| leg | prefix | result |
|---|---|---|
| A, GGUF, 8 models | `sync0342a_` | 8 suites, clean |
| B, MLX, 5 nvfp4 models | `sync0342b_` | 5 suites, clean |
| control, GGUF on the **deployed** build | `depl0342_` | 8 suites, clean |
| repeats, MLX qwen3.8:27b, 3 per build | `candrep0342_` / `deplrep0342_` | 6 suites, clean |

**Against the deployed build, every GGUF cell is identical.** All eight models, every metric:
scene, document, fine text, multi-image, contracts. Nothing in the fold moves a GGUF number.

**The first comparison was against the wrong baseline, and the control is why that is known.**
Compared with `ggml0341_`, gemma4:31b appeared to move two cells — scene 0.966 → 0.963 and
the 9 px tier 4 → 3. Those are [ADR 0036](../adr/0036-gemma4-image-chunk-decodes-in-one-batch.md)'s
own measured numbers, digit for digit: that baseline predates the batch floor and the
candidate contains it, so the comparison measured the batch change. gemma4 e4b, e2b and
nemotron q8 moved against the same stale baseline and are likewise identical to the deployed
build. **A fold's control is the build in production, not the last fold's candidate.**

**MLX: four of five models identical cell for cell**, including gemma4 12b, 26b and 31b —
the models whose entire implementation changed directory in this fold. The 31b leg ran
`gemma4:31b-nvfp4-tower4bit`, the archived four-bit-tower artifact, because production's
`gemma4:31b-nvfp4` was promoted to the library's bf16-tower weights on 2026-09-19 and the tag
no longer names what the baseline measured (ADR 0038). Comparing against the tag would have
invented a regression on the one model most likely to show one.

**The one MLX cell that moved did not move between builds.** qwen3.8:27b name_bbox read
0.541 on `sync0341b_` and 0.484 here. Three repeats on each build: **0.484 on both, identical
across all three runs of each** — zero within-arm spread, so the fold and the deployed build
agree exactly and the difference belongs to the older baseline, not to this fold.

Renders: `preflight-runs/{sync0342a_,sync0342b_,depl0342_,candrep0342_,deplrep0342_}thinkfalse.log`.

## ADR 0039, implemented and verified on the GPU (2026-09-20)

The maintainer's word, "ADR 0039 first, then tag and deploy", so it lands inside this fold, where the
files it touches had just moved.

Global scales are stored as the checkpoint's own `m`; MLX's `m × Nvfp4MaxProduct` form is
built only where MLX consumes it (`GatherQMM`'s native branch and `QQMM`, via
`mlx.ToMLXRepresentation`). `ToMLXGlobalScale` is renamed `LoadGlobalScale`, and the identity
scale is 1 rather than `Nvfp4MaxProduct` — in `GatherQMMIdentityScale` and in dflash's
per-row fill.

**`TestVisionGoldenParity` on the CUDA payload, the measurement the unit tests cannot make:**

| model | before (the fold, pre-0039) | with ADR 0039 | pre-fold 0.33.2 (#312) |
|---|---|---|---|
| 12b (control) | 0.0625 | **0.0625** | 0.0625 |
| 31b, 4-bit tower | 0.1094 | **0.0898** | 0.0898 |

31b reads mean −0.00621, std 1.3243, norm_mean 96.999, max sampled delta 0.0898 — the
pre-fold line in every digit. The encoder is back to bit-parity with `mlx-vlm` and with the
build the goldens were taken from. 12b does not move, which is the control: its two vision
scales survived the old round trip intact.

**The goldens describe an artifact, not a tag.** Run against `gemma4:31b-nvfp4` this test now
fails at max delta 0.7891 — correctly, because production's copy of that tag was promoted to
the library's bf16-tower weights on 2026-09-19 (ADR 0038) and the goldens were taken from the
4-bit tower. The archived `gemma4:31b-nvfp4-tower4bit` (manifest `637cc0ff1570`) is that
artifact, and the test's golden lookup now accepts its suffix. Anyone re-taking these goldens
must say which artifact they used.

**One defect of ours, found by this run and fixed:** the promotion on 2026-09-19 installed the
`31b-nvfp4` manifest and three blobs at mode 0640 while every neighbour in the store is 0644.
Production runs as root and never noticed; an unprivileged reader — this test in its container
— could not open the model at all. Both are 0644 now. The 42 files still at 0600 are ollama's
own `models/metadata`, written that way by the server before this work.

## Deployed, and verified on the build itself (2026-09-20)

Tagged `v0.34.2-dynres` and deployed to `:11497` at 13:08:53 as
`0.34.2-dynres-0-g5bffaac`, eleven seconds without service, 55 tags either side.
ADR 0039 landed inside the fold, as above.

**Everything in gate 6 was measured on `0.34.1-dynres-26-g3dade56`, not on what shipped.**
The deployed image is a Go-only swap onto the same native payload — the only file differing
under `llama/` is `README.md` — so those numbers transfer by argument. An argument is not a
measurement, and ADR 0039 plus two Go changes (the fp16 gate widening to the `qwen2vl`
architecture spelling, the iGPU dedicated-pool fix) sat in the gap with no scored cell over
them. So the suite and the OCRBench ladder were re-run on `maxusai/ollama:sync-0.34.2-main`,
the exact image production serves, on a canary — never `:11497`.

18 suites and 8 OCRBench arms: **0 errors, 0 OOMs, 0 not converged, 0 empty answers**, and
`prompt_eval` identical to the baselines throughout, which is what proves the arms were shown
the same images at the same grids.

### GGUF: nothing moved

**8,172 scored cells, zero moved** — against *both* `0.34.1-dynres-26-g3dade56` (the fold
candidate) and `0.34.1-dynres-16-g16649e8` (the build previously in production). All eight
models, every metric: scene, document, fine text, multi-image, contracts.

### MLX: bounded by the build's own spread

The five nvfp4 models moved 92 of 5,104 cells against the candidate, which means nothing on
its own — MLX think-off is not bit-reproducible on this fork. The control is a **second MLX
leg on the same image** (`prod0342b_` vs `prod0342b2_`):

| comparison | cells moved |
|---|---|
| within one build, run 1 vs run 2 | 40 / 5104 (0.8 %) |
| candidate vs deployed, run 1 | 92 / 5104 (1.8 %) |
| candidate vs deployed, run 2 | 95 / 5104 (1.9 %) |

Same order, same kinds, same cells — and the two largest cross-build flips **flip back** on
the same build:

| cell | candidate | deployed run 1 | deployed run 2 |
|---|---|---|---|
| `gemma4:12b-nvfp4` `bbox_contract` | 0.929, 6 hits | **0.0, 0 hits** | **0.958, 6 hits** |
| `qwen3.6:35b-a3b-nvfp4` `bbox_contract_multi` | 0 hits | **3 hits** | **0 hits** |

A contract dropping from six hits to zero is the most alarming thing in the diff, and one
build produces both answers. Read cross-build alone it is a regression; read against the
within-build control it is prefix-cache state (ADR 0029).

### The one cell that is not noise, and it recovered

`qwen3.8:27b-nvfp4` is nearly deterministic here — **one** cell moved between the two runs,
by 0.001 — yet it shifted 28 against the candidate. The largest is
`document_single.name_bbox_mean_iou`, and its whole history says the same thing:

| builds | runs | `name_bbox_mean_iou` |
|---|---|---|
| 0.33.2 → 0.34.0 | 10 | 0.541 / 0.542 |
| **0.34.1**, both `g16649e8` and `g3dade56` | 6 | **0.484** |
| **0.34.2-dynres-0-g5bffaac** | 2 | **0.542** |

**That is a recovery.** The 0.34.1 fold introduced the global-scale ×2688 / ÷2688 round trip;
ADR 0039 removed it, and the cell returns to its pre-0.34.1 value. It is the same shape the
encoder goldens show — 31b max sampled delta 0.1094 → 0.0898, back to the pre-fold line — so
ADR 0039's benefit is visible end to end in the suite, not only in the goldens.

Gate 6 read 0.484 on both 0.34.1 builds with zero spread across three repeats each and
concluded the difference belonged to the older baseline rather than to the fold. That was
right, and this completes it: 0.484 was a 0.34.1-era regression, not a baseline artefact.

**Attribution is an argument, not an isolation.** The deployed build differs from the
candidate by ADR 0039, the fp16 gate and the iGPU fix; only ADR 0039 touches MLX at all.

### OCRBench: four arms, flat

Rows 0–200, think off, temperature 0, `apply_sampling=False`, one model at a time — the same
protocol as [the CUDA ladder](../ocrbench-quantisation-ladder.md), whose arms were measured on
`0.34.1-dynres-16-g16649e8`.

| arm | `g16649e8` | `g5bffaac` | discordant items | exact McNemar |
|---|---|---|---|---|
| nvfp4 / 4-bit tower | 0.860, 0.860 | 0.865, 0.860 | 1 | inside its own spread |
| nvfp4 / bf16 tower | 0.850, 0.850 | 0.855, 0.855 | 1 | 1.000 |
| GGUF q4_K_M | 0.855, 0.855 | 0.855, 0.855 | 0 | 1.000 |
| GGUF q8_0 | 0.850, 0.850 | 0.850, 0.850 | **0** | 1.000 |

`q8_0` is the strongest row available: both ✓ 170, both ✗ 30, **A only 0, B only 0** — not the
same accuracy but the same verdict on all 200 items, across a payload fold, a retired compat
patch and ADR 0039.

Rendered verbatim by `summarize_extbench.py` (SPEC H7):

| model | scored | errors | empty | correct | accuracy | think | endpoint |
|---|---|---|---|---|---|---|---|
| `gemma4:31b-nvfp4-tower4bit` | 200 | 0 | 0 | 173 | **0.865** | false | generate |
| `gemma4:31b-nvfp4` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `gemma4:31b-it-q4_K_M` | 200 | 0 | 0 | 171 | **0.855** | false | generate |
| `gemma4:31b-it-q8_0` | 200 | 0 | 0 | 170 | **0.85** | false | generate |

ocrbench — `echo840/OCRBench` [test], rows 0..200.

⚠ **MIXED — rows are not one campaign** (hosts: ['http://127.0.0.1:11543', 'http://127.0.0.1:11544']; builds: ['0.34.2-dynres-0-g5bffaac'])

| arm | runs | accuracies | items that changed verdict |
|---|---|---|---|
| nvfp4 tower | 2 | 0.865, 0.860 | 1 |
| bf16 tower | 2 | 0.855, 0.855 | 0 |
| q4_K_M | 2 | 0.855, 0.855 | 0 |
| q8_0 | 2 | 0.850, 0.850 | 0 |

| arm | accuracy | ±1 s.e. | mean s/item | median | mean prompt_eval |
|---|---|---|---|---|---|
| nvfp4 tower | 0.865 | 0.024 | 2.1 | 1.9 | 1115 |
| bf16 tower | 0.855 | 0.025 | 2.1 | 2.0 | 1115 |
| q4_K_M | 0.855 | 0.025 | 5.7 | 5.6 | 1115 |
| q8_0 | 0.850 | 0.025 | 5.2 | 4.9 | 1115 |

| question type | n | nvfp4 tower | bf16 tower | q4_K_M | q8_0 |
|---|---|---|---|---|---|
| Artistic Text Recognition | 50 | 49/50 | 49/50 | 49/50 | 48/50 |
| Handwriting Recognition | 50 | 34/50 | 34/50 | 33/50 | 33/50 |
| Irregular Text Recognition | 50 | 40/50 | 39/50 | 40/50 | 40/50 |
| Regular Text Recognition | 50 | 50/50 | 49/50 | 49/50 | 49/50 |

The MIXED banner is correct and is left standing: the two `q8_0` arms ran in a second
container on `:11544` because they were queued after the first four, so the rows carry two
hosts. One build, two hosts — the guard cannot know which difference matters, and that is the
point of it.

**Timing from this run is not usable and is not reported as a finding.** The baseline's arms
were tight (2.3/2.2 and 1.4/1.5 s/item); these are 2.1/2.4 and 2.1/1.8 — each arm's own two
runs spread 0.3 s/item against the baseline's 0.1. GPU0 carried tritonserver and three other
tenants through the window and their activity cannot be reconstructed. That inflated
within-arm spread is contention's signature, so whether the 4-bit tower still costs its
measured 39 % per image is **unresolved here**; it needs both arms interleaved on a quiet GPU
with `pmon` logged alongside. Accuracy is untouched by this — temperature 0, deterministic
decode, `prompt_eval` identical.

### Not covered

`gemma4:31b-it-bf16`, `31b-mxfp8` and `31b-mlx-bf16` are no longer in the store; re-pulling all
three is ~157 GiB against 217 GiB free on an array at 95 %, so that axis of the ladder was not
re-measured. `31b-mlx-bf16` was refused admission on the previous build anyway (75.0 GiB needed
against 62.7 available with the 16 GiB reserve).

Runs: `preflight-runs/{prod0342a_,prod0342b_,prod0342b2_}thinkfalse.log`,
`preflight-runs/ocrprod-prodocr_*.log`.
