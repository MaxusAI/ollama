# TASK: fold upstream v0.34.4 into main

Upstream [v0.34.4](https://github.com/ollama/ollama/releases/tag/v0.34.4) — fourteen commits, 70 files,
+2,074/−912 — folded on top of `main` at `ba7150428` (#373, which carries the v0.34.3 fold). Branch
`task/upstream-sync-0.34.4`, worktree `claude-scratch/wt-sync0344` on the CUDA host (`ai-server/mlx-cuda`).

**In progress on the CUDA host — please do not start a parallel 0.34.4 fold.** ROCm and Metal: once the merge
lands on this branch, your gate 4 and gate 6 legs can build from it.

## Status (2026-09-24)

| gate | state |
|---|---|
| 1, the merge | **done** — `c3e393d56`; 15 conflicted files. `go build` and `go vet` clean over all 80 packages; `go test` 58 packages ok, 0 failed; `server` green with `OLLAMA_FORMAT_TWO_PASS` unset **and** set. MLX tests skip here until gate 4 builds the payload |
| 2, docs and paths | **done** — `check_source_paths.py` clean over the 66 files the fold wrote |
| 3, the patch series | **done** — all seven (001 002 004 005 801 802 903) apply clean to `b11081` on a real checkout, in order; served projectors unchanged |
| 4, image | next — **a full build**: native inputs move, so a Go-only swap is not valid |
| 5, preflight | pending — `payload_pin` and `mlx_payload_pin` will fail by design until the pins move with evidence |
| 6, campaigns | pending — control is production, `0.34.2-dynres-0-g5bffaac`, whose suite and OCRBench were re-measured on 2026-09-20 |

**Three hosts converged on this merge.** The ROCm and Metal hosts had each started the same fold before #375
existed, stopped, and cross-checked instead; see [#375](https://github.com/MaxusAI/ollama/pull/375).

## What v0.34.4 changes for the fork

- **llama.cpp `b10969` → `b11081`.** 112 builds. Gate 3 applies the whole series (001, 002, 004, 005, 801, 802,
  903) with plain `git apply` on a real checkout first, and diffs the projector cases of every served model.
- **MLX `d9add9d1` → `59d600b5`**, the first MLX move since `d9add9d1`. The vision goldens are re-taken against
  the archived artifacts, and the kernel fixes in the range are checked against the served models' dimensions.
- **XGrammar 0.2.5 → 0.2.7.** 0.2.5's `allOf` with more than one branch enforces nothing; re-checked on 0.2.7.
- **Structured outputs after thinking** (four commits across `server/`, `llm/`, `mlxrunner/`, `model/parsers/`)
  — meets the fork's ADR 0004 marker flow and ADR 0033 grammar passthrough.
- **`mlx: select Gemma 4 image resolution dynamically`** (#18603) — a policy change, see below.
- **`mlx: speed up Qwen 3.8 prompt processing`** (#18550) — a fused, scale-deferring SwiGLU, see below.

## Resolved so far

### gemma4 on MLX: the fork's pipeline, unchanged

Upstream replaces the checkpoint's fixed budget with a per-image choice among 70/140/280/560/1120 "closest to
the input resolution … without adding an API parameter". The fork's contract is the opposite shape: per-request
`image_min_tokens`/`image_max_tokens`, defaults 70/1120, **fill** the budget and snap to the ladder, shared with
the GGUF path through `llm.BudgetFillSize` (ADR 0003, 0008, 0021). ADR 0008 exists because under-filled gemma4
grids measurably lost grounding. So the seven gemma4 files resolve to the fork's pipeline — byte-identical to
`main` — and upstream's `process_image.go` stays deleted.

Upstream's commit also adds a guard that rejects a grid overrunning the learned position table. On every served
gemma4 checkpoint the table is 10,240 per axis and the fork's 1,120-token ceiling bounds the longest side at
1,120 × 3 = 3,360 patches, so the guard is unreachable here. **Adopting upstream's per-image budget as a default
is a policy decision for an ADR with a measurement, not a merge resolution**; it is not made in this fold.

### `mlx/ops_extra.go`: upstream's scale helpers, in ADR 0039's terms

Upstream still stores global scales in MLX's `m × 2688` form, and #18550 adds `globalScaleFactor(s) = s / 2688`
and `identityGlobalScale() = 2688`, used by `scaleAndCast` and by a new fused `SwiGLUScaled` that defers each
projection's scale out of the matmul. This fork stores the checkpoint's own `m` (ADR 0039). **The conflicted line
was the easy part: `mlx/act.go`'s call to `globalScaleFactor` auto-merged cleanly**, and under `m`-valued scales
it would scale every deferred nvfp4 gate and up projection by 1/2688 — and compile.

Resolved by defining the two helpers in ADR 0039's terms — the stored `m` already is the multiplier, the identity
is 1 — so every upstream call site is correct as written. Upstream's two new tests could not see this: each
compares two paths that share `globalScaleFactor`, so they agree under either representation. Their fixtures now
use the stored form, and `TestSwiGLUScaledMatchesSeparateScaling` gains an assertion that applies the factor
directly, which is the check that fails if the division comes back.

### Think+format: single pass by default, two-pass behind a switch

Upstream replaced the two-pass structured-output flow with a single pass on both engines: the parsers report the
strings that end their thinking (`ThinkingClose`), the server names them on one completion request, and each runner
constrains only what follows — llama-server through a GBNF wrapper around its own schema conversion, MLX through an
XGrammar structural tag. It removes the second prefill, the dropped boundary chunk, and MLX's stray first token in the
JSON, and it covers all 18 parsers where the fork's marker hook covered two.

**The maintainer's decision: single pass is the default, and `OLLAMA_FORMAT_TWO_PASS=1` keeps ADR 0004's flow as the rollback**
if single pass regresses on a served model.

- **`routes.go` is resolved by function, not by hunk.** Both handlers were rewritten too deeply: taking the fork's
  side of each hunk left upstream's deletions *between* the hunks, including the `structuredOutputsState` type the
  kept code uses. So the file is `main`'s handlers plus upstream's two changes outside them — `getExistingName`
  (#18438) and the `thinkingCloseForCompletion` helper — and the switch gates the fork's own defer decision
  (`deferViaMarker`/`deferViaTransition` in Generate, `deferring` in Chat). Closing strings go on the request only
  when it is off; with it on, pass two's prompt already ends past the marker.
- **The two-pass hooks come back in the lower layers.** Upstream deleted `IncludeIntermediateMetrics` from
  `llm.CompletionRequest`, llama-server's `TimingsPerToken` and per-chunk metrics, and the MLX request literal —
  every one in a file that merged without a conflict. All are restored and inert unless the switch is on.
- **llama-server:** upstream's block landed inside the fork's `runCompletionPhase`, which returns `(result, err)`,
  and returned a bare `err`. The ROCm host flagged this before it bit.
- **MLX:** upstream's thinking-aware structural tag wraps the fork's whitespace-bounded `json_schema` element, so the
  bound holds on both branches; the plain tag is byte-identical to `main`'s.
- **Tests pin the mode they test.** The nine two-pass tests set `OLLAMA_FORMAT_TWO_PASS=1`; they were found by
  running each candidate alone under the default (eight fail, one hangs waiting for a second request). The
  single-pass route tests, taken from the ROCm host's cross-check, pin it unset. With the switch on they fail with
  "got 2 completion calls, want 1", which is the switch visibly changing the flow.

**What single pass changes, for the gates** (from the ROCm host's review):

1. `num_predict` now bounds the total output; the two-pass total could exceed it (8290 vs 8192 on qwen3.6
   `bbox_contract_reasoning`).
2. With format, tools and thinking together, a tool call can no longer replace the formatted answer (harmony excepted).
3. Raw generate never defers: the format applies from token 0.
4. An EOS inside the thinking returns only the thinking, `response:""`, `done_reason:"stop"`.
5. **On MLX a think+format request carries a grammar from its first token**, so with production's
   `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` it never drafts during the thinking, where pass one used to. Measure at 0 and 1.

Think-off cells cannot tell the two flows apart — the suite always sends `format:"json"` and both constrain from
token 0 when thinking is off — so gate 6 needs the think-on `bbox_contract_reasoning` cells. nemotron3 has no
sampling card, so its think-on cells run at packaged defaults and compare as rates, not cells.

## Cross-checks from the other hosts

| host | tree | result |
|---|---|---|
| ROCm, gfx1151 | its own merge, `dd19f1202` | gemma4 and `ops_extra.go` resolved identically; `requestGrammar` byte-identical. b11081 and all seven patches build on HIP (rocm7 and rocm10). Single pass end to end on llama-server: gemma4:e2b and qwen3:0.6b 5/5; the grammar is transparent to the thinking in matched cache state |
| Metal, M5 Max | the ROCm tree | **MLX tests where they execute: 900 passed, 0 failed, 4 stated skips.** XGrammar 0.2.7's standalone CMake builds clean on macOS. The ADR 0039 helper fix reached a third time; against upstream's helpers the contract test reads `SwiGLUScaled()[0] = -5.6e-07, want -0.0122` while upstream's own test passes all five cases |

One correction between the readings: the ROCm review lists `tools/mtmd` as unchanged in the range. `git diff
b10969 b11081 -- tools/mtmd/` shows one hunk, `clip.cpp` checking that the compute graph allocated. Its conclusion
stands, since that is error handling rather than preprocessing.

## Found on `main`, not caused by this fold

Sweeping the merged tree for the old representation found **three sites ADR 0039 missed**, all from upstream's MLX
bump of 2026-09-14, six days before ADR 0039 landed:

| site | what it does | under ADR 0039 |
|---|---|---|
| `mlxrunner/model/laguna/laguna.go:731` | divides a `QuantizedLinear.GlobalScale` by 2688 | that field holds `m`, so dense laguna nvfp4 output is ×1/2688 |
| `mlxrunner/model/nemotron_h/nemotron_h.go:447` | divides the loader's scale by 2688, "to undo MLX's representation" | `ReadGlobalScale` returns `LoadGlobalScale(found)` = `m`; experts are ×1/2688 |
| `mlxrunner/model/qwen3_5/gdn_projections.go:195` | fills a missing scale with 2688 "in MLX's representation" beside present ones in `m` | a mixed bank; reachable only when one of a hi/lo pair lacks a scale |

**None is on a model production serves**, and production's own campaign agrees: qwen3.8:27b-nvfp4 runs through
`gdn_projections.go` and scored 0.99 IoU, so its pairs always carry both scales. `TestApplyExpertWeightGlobalScale`
passes because its fixture feeds the MLX form the loader no longer produces. These get their own change with a
representation-sensitive test each, so the fold's attribution stays clean.
