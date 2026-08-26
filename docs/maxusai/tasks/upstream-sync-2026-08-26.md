# TASK: merge upstream v0.33.0 (ebf200f9) into main

**Opened:** 2026-08-26. **Status:** merge PERFORMED on this branch (zero
conflicts); local test gates run; the Metal-host gates and the image
build + preflight remain, listed under acceptance criteria. The build is
deliberately deferred — the operator paused rebuilds; vsuite carries the
interim `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` workaround until a build from
this lineage replaces it.

## Scope

Merge the upstream release tag `v0.33.0` (`ebf200f9`) into `main`.
Merge-base is `8f912415` (v0.32.15-6-g8f912415 — the #208 sync executed
against a slightly advanced upstream/main, so the fork base sits 3 commits
past that task's stated target). Exactly **20 upstream commits** are in
scope, authored 2026-08-21 → 2026-08-25. A dry-run
`git merge-tree --write-tree main v0.33.0` (run 2026-08-26, with the merged
qwen25vl gate and preflight poison probe on main) produced a clean tree —
**zero conflicts**, versus two in the #208 sync.

**Both payload pins are unchanged:** `LLAMA_CPP_VERSION` stays `b10488`
(= `9d77fa172`) and the MLX pin stays `27fec909…`. Consequences, each
load-bearing for the effort estimate:

- All `llama/compat/` patches (001/002/003/004/005/903) carry **unmodified**
  — clean-room validated against exactly this pin on 2026-08-26. 903 stays
  required (ggml-org/llama.cpp#27044 still unfixed at this pin).
- Preflight `payload_pin` and every recorded expectation remain valid —
  **no re-measurement**. The `poison_probe` canary carries as-is.
- Fork versioning is fork-tag-derived (`git describe` → `0.32.14-dynres-…`),
  so the `cuda-dynres-903` `version_pattern` keeps matching. If a new fork
  tag is cut for the 0.33.0 lineage, widen the pattern and note it in the
  profile.

## The 20 commits

| Group | Commits | Assessment |
|---|---|---|
| Claude Desktop / app / launch / proxy | `30546d1f`, `5ad1681c`, `30019c87`, `2d9622a4`, `124e9af9`, `add1f92b`, `fb307609`, `93942515`, `e2e82903`, `60d83f8b`, `82ad9fa3`, `f6c59d87`, `6e19e916`, `377ef091`, `075aa7e1` | New, self-contained subsystem (`internal/proxy/claude_desktop*`, ~3.8k lines incl. tests) plus `app/`/`cmd/launch/` churn. Inert to the serving image; auto-merges as new files. |
| **mlxrunner prefix-cache robustness** | `c01eafa5`, `b315b3ee`, `30e28918`, `81f9a394`, `c44575ef` | **The substance.** Prefill snapshots kept on mid-prompt cancel, prefill captures clipped to trie-node edges, generated-token page-out on close, trie growth by whole child nodes so restore points survive resumed prefills, draft-cache settling on cancelled prefills. Directly relevant to fork MLX campaigns (long prompts, cancellations, speculative drafts). |
| `ebf200f9` | proxy: preserve string content during image fallback | Small, server-side; v0.33.0 tag commit. |
| `02dc3ea4` | cmd: guard empty editor before indexing fields | Trivial. |

## Overlap analysis (why zero conflicts still needs review)

The fork's delta vs the upstream base intersects upstream's delta in exactly
two files: `x/mlxrunner/pipeline.go` and `x/mlxrunner/prefix_cache.go`. Both
auto-merge textually. The fork carries +4.4k lines of its own mlxrunner
layer (vision goldens, e2e, stopper tests) around them, so the gate for
*semantic* compatibility is the fork's own mlx test suite and the
12b/26b/31b vision golden tests — the same rule as #208 (single-owner-thread
applies).

Follow-up owed after this lands: `fix/mlx-thrash-check-default-off` also
touches `pipeline.go` (upstream's delta there is +4 lines) — re-merge that
branch on top.

## What this merge does NOT change

- **The qwen2.5vl fp16-accumulate class is alive in stock 0.33.0**
  (measured, #214 / #216) — the merged launcher gate stays load-bearing and
  `poison_probe` asserts it on every preflight.
- The stock-0.33.0 sticky-slot *recovery* is not in this window's server
  code (upstream touches nothing under `llm/`, `server/` scheduling, `ml/`,
  `llama/` except `model_recommendations*`); it most likely arrived with the
  already-merged `e0c95a5f` parser-deadlock fix. One poison-probe cell on
  the built image settles it — and with the gate, poisoning has nothing to
  stick to.

## Acceptance criteria (in order)

1. ✅ Merge on this branch; zero conflicts (dry-run and actual).
2. ✅ `go test ./server/ ./model/renderers/ ./model/parsers/ ./llm/` green in
   the `golang:1.26` container (`-u 1000:1000`, `-buildvcs=false`; the
   fileutil root-caveat and the app/dist embed baseline from #208 apply).
3. ☐ mlx suite + vision golden tests (`x/mlxrunner/vision_golden_test.go`,
   12b/26b/31b) on the Metal host — the semantic gate for the two
   overlapping files.
4. ☐ Image build (bigdisk builder; GOFLAGS version stamp per
   `scripts/env.sh`; expect the MLX-stage cache miss ≈ 3 h nvcc) + full
   preflight on the CUDA host (`poison_probe` included) + `mlx-metal`
   preflight on the Apple host, since the in-scope changes are mlx-runtime
   behavior. Deferred until the operator green-lights the rebuild.
5. ✅ Merge-commit body notes SPEC H11 `server_version` as the comparability
   boundary for benchmark cells measured on builds from this merge, and
   that vsuite's interim global-f32 workaround retires when this deploys.
