# TASK: fold upstream v0.40.3 into main

Upstream [v0.40.3](https://github.com/ollama/ollama/releases/tag/v0.40.3) (2026-10-11) folds on top of `main` at
`377f9d144` (#487), which carries the v0.40.2 fold (#460) and the `v0.40.2-dynres.1` point tag.
- **Size:** v0.40.2..v0.40.3 is 3 commits, 6 files, +61/−11. All Go: no native input moves. llama.cpp stays b11351
  (`631109b34`) and MLX stays `a59cc231`.
- **Where:** branch `task/upstream-sync-0.40.3`, worktree `.claude/worktrees/fold-0403` on the ROCm host
  (`amd-server/rocm-gfx1151`), at the maintainer's word (2026-10-13: "sync MaxusAI/ollama upstream").
- **Not in scope:** v0.40.4 is at `rc1` (2026-10-12): 25 commits, 76 files, and llama.cpp moves to b11521. It is a
  separate fold with a native move, once it is released.

## What v0.40.3 changes for the fork

- **The local GGUF compatibility migration is paused** (`eab97e9f9`, upstream #18908).
  - v0.40.2 converts a legacy GGUF into a `llamacpp` child on its first load and writes the child into the store as
    a second full copy (`compatmigrate`, `75b952780`).
  - v0.40.3 comments out the call in `scheduleRunnerForModel` ("TODO: enable compatibility migration") and skips
    `TestLocalCompatibilityMigration`. `compatmigrate` stays in the tree, because `server/create.go` still uses it.
  - **For the fork's hosts, a v0.40.3 deploy no longer rewrites production's store.** A v0.40.2 deploy converts each
    legacy GGUF on its first load, about 1.25x its size in free space while converting. On gfx1151, with about 90 GB
    free against an 830 GB store, that was the reason the `v0.40.2-dynres.1` deploy was deferred.
  - **Children already converted by a v0.40.2 deploy stay.** v0.40.3 still resolves a tag's `llamacpp` child, so a
    host that served v0.40.2 keeps serving the converted copies. Only new conversions stop.
  - **#483 is still needed.** A model pulled from the registry already in the `llamacpp` layout is
    `nemotron_h_moe` plus a projector without any conversion, and #483's `VisionBudgetArch` covers it. On a legacy
    nemotron3 GGUF the arch stays `nemotron_h_omni`, as on 0.35.0.
- **Launch:** Claude Code's connector warning and Codex's inherited service-tier warnings are suppressed
  (`021cda522`, `c2b7368d4`). No fork code touches `cmd/launch`.

## Status

| gate | state |
|---|---|
| 1, the merge | **done**: `fd14dc60b`, no conflicts. `go build ./...` is clean; `go vet` and `go test` pass for `./server`, `./cmd/launch`, `./llm` and `./compatmigrate` |
| 2, docs and paths | this record |
| 3, the patch series | unchanged: no native input moves, so the series that applies at b11351 (#462's 903) is the one that ships |
| 4, image | **gfx1151**: `maxusai-ollama:0.40.2-dynres.1-6-g9f3baaf-rocm7-gfx1151`. Its 19 native libraries are byte-identical to the `v0.40.2-dynres.1` release's. **CUDA**: built (#488) |
| 5, preflight | **gfx1151: VERDICT PASS, PASS=27 SKIP=5** against `rocm7-0-40-2` (`runs/preflight-rocm7-0403-fold-g9f3baaf.json`), the shape of the `v0.40.2-dynres.1` release run, in production's environment. The server log holds no migration line, so every model ran as its legacy GGUF; nemotron's pinned budget is 3328 -> 3270 and `payload_proof` is custom on max/min. **CUDA: both preflights pass** (#488). **Metal**: open. The fold's interim stamp `0.40.2-dynres.1-N-g<sha>` resolves to the 0.40.2 profiles on every platform; `rocm7-0-40-2` is widened here to admit the `v0.40.3-dynres` lineage (ADR 0032: a Go-only fold shares its payload's profile, as the v0.35.0 fold did) |
| 6, campaigns | not needed for a Go-only fold that touches no inference path; each host's call |
| tag and deploy | not cut, not deployed |
