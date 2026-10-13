# TASK: a converted nemotron3 GGUF keeps its per-request image budget (#480)

Claimed by the Metal host (`macbook-pro-m5-max-128GB/mlx-metal`) on 2026-10-13. Found by the ROCm host's preflight (#480, #481).

## The defect
v0.40.2 converts nemotron3's legacy GGUF on first load (`compatmigrate/nemotron3.go`). The child is a text model with
`general.architecture = nemotron_h_moe` plus a separate `clip` projector (`nemotron_v2_vl`), and its config reads
`"model_family":"nemotron_h_moe","model_families":["nemotron_h_moe","clip"]`. Every fork switch that gives nemotron its image
budget matches `nemotron_h_omni` only, so a converted model:
- launches with no `--image-min-tokens` / `--image-max-tokens` (`visionServerArgs`);
- never reloads on a budget change (`ResolvedImageTokenBudget` / `NormalizeImageTokenBudget`, used by `server/sched.go`);
- falls back to `defaultImageTokensEstimate` in prompt truncation (`MaxImageTokens`, `ImageTokensForSize` via `visionTokenArch`);
- fails the nemotron capability check (`server/images.go`).

Measured on gfx1151: `image_max_tokens: 3328` on a 2048×1152 image gives 2306 tokens (the unpinned control) on 0.40.2, against
3270 on 0.35.0.

## The plan
Resolve a "vision arch" once: `nemotron_h_moe` **with** a projector is `nemotron_h_omni`, and use it everywhere above. The
text-only Nemotron 3 Nano (`nemotron_h_moe`, no projector) must stay unaffected. Test-first; touch as little of the budget
code as possible, because #468 rewrites the same functions.

## Done when
- unit tests pin both shapes (converted child with projector: budget honoured, reload on change, image cost; text-only Nano:
  unchanged) and fail on main;
- the PR names the merge order against #468;
- a live check (pinned budget on a converted nemotron3 GGUF) runs on a host with GPU time to spare.
