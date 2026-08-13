# ADR 0022: thinking stays off for vision work; gemma4 is the only lineage where it is even viable

- **Status:** accepted (2026-08-13)
- **Date:** 2026-08-13
- **Deciders:** MaxusAI fork maintainers
- **Related:** [campaign 2026-08-13](../vision-campaign-2026-08-13-thinkmode.md) (the
  measurements this rests on), [ADR 0012](0012-benchmark-report-templates.md) (report
  templates and the ±0.01 noise floor), [ADR 0011](0011-preflight-expectations-are-versioned-code.md)
  (`min_num_predict`, the first of the traps below),
  [ADR 0004](0004-routes-layer-think-format-double-request.md) (think+format at the routes
  layer), [ADR 0001](0001-nemotron-vision-dynamic-resolution.md) (nemotron dyn-res)

## Context

`think` was never measured against the vision suite. The assumption worth testing was the
intuitive one — that letting a model reason before answering should improve grounding,
particularly on bounding boxes. It does not.

Measured 2026-08-13 on gfx1151/ROCm, server `0.32.1-dynres-296eb020` (`v0.32.1-dynres.3`,
payload b9888 + compat 001/002/004/005), `/api/chat`, temperature 0, `num_ctx = 65536`.
Full matrix and provenance in the campaign; the load-bearing numbers:

| model | metric | off | on | Δ |
|---|---|---|---|---|
| nemotron3:33b | scene IoU | 0.840 | **0.391** | **−0.449** |
| nemotron3:33b | document items | 5/5 | **4/5** | **−1** |
| gemma4:26b-a4b | document IoU | 0.810 | **0.718** | **−0.092** |
| gemma4:31b | document IoU | 0.760 | **0.721** | **−0.039** |
| qwen3.6:35b-a3b | scene IoU | 0.953 | 0.962 | +0.009 *(noise)* |
| qwen3.6:35b-a3b | multi-image | ✅ | ❌ **non-terminating** | — |

Every delta that clears ADR 0012's ±0.01 noise floor is **negative**. Nothing improves.

Two failure modes are qualitatively worse than "no benefit":

**nemotron3 loses spatial precision without losing comprehension.** Labels, serial, invoice
number and total all stay correct while scene IoU more than halves and an extraction item
disappears. The model appears to reason its way off coordinates it would otherwise read
directly.

**qwen3.6 does not terminate on multi-image cross-referencing.** Budgets of 12,000, 32,000
and 64,000 were each consumed exactly, with no partial convergence. Captured thinking has a
unique/total token ratio of 0.211 with 15-grams repeating ×3 — it re-enumerates the same
cross-image word list indefinitely against a prompt that asks it to be "exhaustive". The
same model on a generic three-image prompt finishes in 2,113 tokens with non-repeating
reasoning, so this is prompt-specific rather than a model or build defect. The hard ceiling
is `262,144 − 6,134 = 256,010`, so no budget rescues it.

## Decision

**Vision requests are served with `think` off.** This is the default and it stays the
default; it is not a per-request preference to tune for quality.

1. **`nemotron3` — never.** Grounding collapse plus extraction loss.
2. **`qwen3.6` — never.** Non-terminating on multi-image; elsewhere it buys +0.009 IoU for
   19,160 tokens (~5.5 min at 57 tok/s against ~15 s).
3. **`gemma4` — permitted, but it buys nothing.** The only family that completes every test
   (2.0k–6.0k tokens, ~10× shorter than qwen3.6) and the only one that passes multi-image
   with thinking on. Enable it when the *reasoning trace* is the deliverable, never to
   improve vision output — it still costs 0.04–0.09 document IoU.

**A think-on regression claim is not admissible without checking `eval_count` against
`num_predict` first.** See the consequences below.

## Consequences

- Benchmarks and deployments keep `think: false` for vision. The campaign is the reference
  if this is revisited; re-derive nothing from intuition.
- Anyone wanting a reasoning trace on vision uses gemma4 and accepts the grounding cost.
- **Three harness traps are now documented, because each fakes a regression.** They are
  properties of measuring think-on at all, not of this build:
  1. **`eval_count == num_predict` exactly** is budget exhaustion inside an unclosed
     thinking block, not a vision failure. The first run of this campaign returned
     `json_valid=false` and IoU 0.0 on every vision test and would have been reported as
     "think-on breaks vision on this build". `expectations.toml`'s `min_num_predict`
     (ADR 0011) documents this at the ~600 floor; it recurs at *every* budget below the
     model's real thinking length.
  2. **Raising `NUM_PREDICT` without `NUM_CTX`** converts truncation into a hard 400 — the
     server checks `prompt + num_predict <= num_ctx` up front (`2715ffce`, `18dc2e63`).
  3. **`THINK` must be the literal string `"on"`.** `run_engine_compare.sh` defaults it to
     `"false"` and the suite tests `== "on"`, so `THINK=true` silently benchmarks with
     thinking **off** — a think-mode result that never enabled thinking.
- Not measured, and deliberately not extrapolated: nemotron3 multi-image with think-on
  (both measured arms regressed, so it was not pursued), and fine-text beyond qwen3.6.

## Alternatives considered

- **Enable thinking selectively for hard grounding tasks.** Rejected: document grounding is
  exactly where it regresses most on gemma4, and scene grounding is where nemotron3
  collapses. The tasks that would motivate it are the ones it damages.
- **Raise budgets until qwen3.6's multi-image terminates.** Rejected on evidence: three
  escalations consumed their budget exactly with no partial convergence, and the repetition
  signature identifies a loop rather than slow reasoning. The remaining rungs cost ~50 min
  and ~2 h for a near-certain repeat, and 256,010 is a hard ceiling.
