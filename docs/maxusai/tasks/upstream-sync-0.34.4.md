# TASK: fold upstream v0.34.4 into main

Upstream [v0.34.4](https://github.com/ollama/ollama/releases/tag/v0.34.4) — fourteen commits, 70 files,
+2,074/−912 — folded on top of `main` at `ba7150428` (#373, which carries the v0.34.3 fold). Branch
`task/upstream-sync-0.34.4`, worktree `claude-scratch/wt-sync0344` on the CUDA host (`ai-server/mlx-cuda`).

**In progress on the CUDA host — please do not start a parallel 0.34.4 fold.** ROCm and Metal: once the merge
lands on this branch, your gate 4 and gate 6 legs can build from it.

## Status (2026-09-24)

| gate | state |
|---|---|
| 1, the merge | **in progress** — 15 conflicted files; the gemma4 cluster and `mlx/ops_extra.go` are resolved, the structured-output cluster is next |
| 2, docs and paths | not started |
| 3, the patch series | not started — **both pins move**, so every `llama/compat` patch is checked on a `b11081` checkout before any build |
| 4, image | not started — **a full build**: native inputs move, so a Go-only swap is not valid |
| 5, preflight | not started — `payload_pin` and `mlx_payload_pin` will fail by design until the pins move with evidence |
| 6, campaigns | not started — control is production, `0.34.2-dynres-0-g5bffaac`, whose suite and OCRBench were re-measured on 2026-09-20 |

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
