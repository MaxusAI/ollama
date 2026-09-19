# TASK: fold upstream v0.34.2 into main

Branch `task/upstream-sync-0.34.2`, worktree `claude-scratch/wt-sync0342`.

## Status (2026-09-19)

| gate | state |
|---|---|
| 1, the merge | **done** — `go build` clean, `go vet` clean, `go test` 57 packages ok |
| 2, docs and paths | **done** — 53 files re-pointed, link check clean |
| 3, the patch series against b10969 | **done** — 6 of 7 apply clean; 906 retired as obsolete |
| 4, image build | **done** — `maxusai/ollama:sync-0.34.2`, 2 h 16 m, rc=0 |
| 5, preflight | **PASS 21 / SKIP 4** on a canary, after the payload pin moved with evidence |
| 6, campaigns | pending |

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

## Next

Gate 6, the campaigns: GGUF and MLX cells against the deployed build, per ADR 0012. Then
Glenn's calls — the release tag, and whether ADR 0039 (the nvfp4 global-scale
representation) lands before or after the deploy.
