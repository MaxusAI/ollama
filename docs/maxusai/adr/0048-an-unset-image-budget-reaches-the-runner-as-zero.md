# ADR 0048: An unset image budget reaches the runner as zero

Date: 2026-10-11 · Status: proposed. Supersedes the "sentinel" parts of
[ADR 0016](0016-reload-on-resolved-vision-flags.md) (its Consequences: "an explicit `70/1120` on nemotron is still
read as unset and is not expressible") and [ADR 0021](0021-gemma4-serves-vision-through-upstreams-media-model.md)
(decision 3: "a value equal to the shared api default counts as unset"), and the matching rows of
[nemotron-dynres-patch.md](../nemotron-dynres-patch.md).

## Context

`api.DefaultOptions` fills `image_min_tokens` / `image_max_tokens` with gemma4's ladder, 70/1120
([ADR 0008](0008-gemma4-budget-fill-restores-1120.md)), so every request reached the runners carrying a budget whether or
not the caller set one. A model whose own bounds differ could not tell the two apart. Two resolvers therefore read
a value equal to 70/1120 as "unset" and substituted the model's own:
- `nemotronImageTokenBudget`, on the llama.cpp path;
- `ResolveImageBudget`, on MLX, which covers muse-glimmer, qwen3.5 and nemotron_h.

ADR 0016 kept this knowingly: an explicit 70 or 1120 on nemotron "is not expressible — pick an adjacent value". The
resolver comment named the fix: the routes layer's `hasOption` check could tell them apart "if that ever matters".

It matters now. On muse-glimmer, whose ceiling is 4096, 1120 is the value a client is most likely to send, because it
is the documented default. Measured on `muse-glimmer:30b-nvfp4` and `30b-mxfp8`, MLX, 2026-10-11 (#463), on a
2048×2048 image:

| `image_max_tokens` | unset | 1024 | 1119 | **1120** | 1121 | 2048 |
|---|---|---|---|---|---|---|
| image tokens | 4100 | 1025 | 1091 | **4100** | 1091 | 2022 |

A request for 1120 cost 3.7 times what it asked for, and nothing reported it.

## Decision

1. **The server sends an unset budget as 0.** `modelOptionsWithEmbeddingBatchDefault` records whether the request or
   the Modelfile set each bound (`hasOption`, as it already does for `draft_num_predict`), and zeroes each one that
   neither set. That holds only when a model is resolved; a nil model keeps the api defaults.
2. **The resolvers read only `<= 0` as unset.** `nemotronImageTokenBudget` and `ResolveImageBudget` lose their
   equality test against the api defaults. Any explicit value is honoured, 70 and 1120 included.
3. **gemma4 is unchanged.** Its resolvers already map `<= 0` to 70/1120, its own bounds. An unset request and an
   explicit 70/1120 launch the same flags and do not reload each other (`TestSchedNeedsReloadImageTokenBudget`).

## Consequences

- **An explicit 70 or 1120 is now a budget like any other** on nemotron_h (both paths), muse-glimmer and qwen3.5
  (MLX). On nemotron it launches different flags than an unset request, so the scheduler reloads for it, as for any
  other change of budget (ADR 0016's B3).
- **Unset behaves exactly as before on every arch.** It still resolves to each model's own bounds. The preflight
  rows send no budget except the pinned probe, whose values (560 on gemma4, 3328 on nemotron) were never the
  sentinel, so no expectation moves.
- **Clients that sent exactly 70 or 1120 to nemotron or glimmer expecting the model's default now get what they
  asked for.** No harness in this repo does: the vision suite sends a budget only when `IMAGE_*_TOKENS` is set.
- **One rule now stands for "unset" across the fork:** `draft_num_predict` on MLX (-1) and the image budget (0) are
  both decided at the routes layer, where the request's raw options are still visible. A runner never infers it from a
  value.
