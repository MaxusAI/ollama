# Release record — `v0.32.1-dynres.3` (296eb020)

Candidate for the third ROCm release on this lineage, superseding
[`v0.32.1-dynres.2`](https://github.com/MaxusAI/ollama/releases/tag/v0.32.1-dynres.2)
(`258534eb`, 2026-08-02).

## Provenance

Per [ADR 0012](https://github.com/MaxusAI/ollama/blob/main/docs/maxusai/adr/0012-benchmark-report-templates.md) (on `main`) every report carries this header.

| | |
|---|---|
| Date | 2026-08-13 |
| Host | Ryzen AI Max+ 395 / Radeon 8060S (**gfx1151**), 96 GiB VRAM, Linux, ROCm |
| Power mode | `n/a` (Linux; `pmset` is macOS-only) |
| Server | `0.32.1-dynres-296eb020` · payload **b9888** · patchset **001+002+004+005** |
| Image | `maxusai-ollama:0.32.1-rocm-dynres-296eb020` (3.08 GB, full `FLAVOR=rocm` build) |
| Endpoint | `chat`, `think: false`, temperature 0 |
| Serving | `OLLAMA_NUM_PARALLEL=1`, cold server per model, model store mounted read-only |

> The release **number** is a git tag; the **version string** stays
> `0.32.1-dynres-<sha>`. Preflight resolves its profile from the version against
> `^0\.32\.1-dynres-[0-9a-f]{7,40}$`, so a version of `0.32.1-dynres.3-<sha>` makes it
> refuse with exit 2 (observed 2026-08-13). Tag the release, not the binary.

## What it contains

29 commits since `.2`; +1,339/−83 lines of Go. The substantive ones:

| commit | change |
|---|---|
| `c138bcd2` ⚠ breaking | charge **real** per-image context cost. The flat-768 heuristic charged `nemotron_h_omni` **zero** (inline vision, empty `ProjectorPaths`) and under-charged `gemma4`/`qwen35moe`, so a chat could pass the Go-side fit check and then overflow |
| `a3acdfdb` | charge images on the native truncation path too (B9) |
| `1c59def5` | transition-flow metrics count both passes with image-aware prompt costs (R6) |
| `0a2c2d70` | scheduler compares **resolved** vision flags, not raw image-token options (ADR 0016) |
| `593fc3b1` | compat **005** — a pinned budget never exceeds the ceiling |
| `ed22deaf` | preflight regression harness backported from main |
| `bc6e06d3` | measured `rocm-0-32-1-dynres` preflight baselines on gfx1151 |

It also closes a gap: production had been running `e74f16d8`, which sits between `.2` and
this head, so the deployed artefact corresponded to **no release tag**.

## Gate status

Still compliant with [amd-upgrade-gate.md](https://github.com/MaxusAI/ollama/blob/main/docs/maxusai/amd-upgrade-gate.md) (on `main`) — this is a move *within*
the b9888 lineage, not past it:

- payload **b9888** (unchanged)
- `dio in binary: 0` — no `--direct-io` **and** no `--load-mode dio`
- `scripts/dio-gate.sh` (deployment repo) passes on `296eb020`

⚠ **Upstream renamed the flag.** `--direct-io` became `--load-mode dio` in `c82ebbd5`,
under the identical `linux && integrated && ROCm` condition. A gate matching only the old
literal reports OK for a tree that still forces direct I/O. Any checker must match both
spellings and fall back to detecting the condition itself.

## Preflight — **PASS**

```
PASS  version · image_tag · go_patch_marker
PASS  text_baseline   [nemotron] in-image prefix 20; text-only 21 is +1 and MUST NOT be used (B8)
PASS  token_ladder    [nemotron] 5/5 within ±2
PASS  payload_proof   [nemotron] min 256*32²=262144, max 3328*32²=3407872
PASS  think_format    [nemotron] valid JSON after thinking (365 tokens)
PASS  pinned_budget   [nemotron] pinned 3328 -> 3270 (ceiling 3328)
PASS  text_baseline   [gemma4]   prefix 19; text-only 19
PASS  token_ladder    [gemma4]   5/5 within ±2
PASS  payload_proof   [gemma4]   min 70*48²=161280, max 1120*48²=2580480
PASS  think_format    [gemma4]   valid JSON after thinking (165 tokens)
skip  pinned_budget   [gemma4]   no expectation recorded for this arch (deliberate)
PASS  endpoint_exclusive        no contention (worst queue wait 0.0 s)

PASS=13  SKIP=1        VERDICT: PASS
```

Identical to the `e74f16d8` run.

## T1 — Campaign matrix

Rendered by `vision-suite/summarize_engine_compare.py` from `scores_*.json` + `ft_*.json`
(ADR 0012: harness tables are generated, never typed). The generators live on `main`; this
lineage carries `preflight/` only, so the output is reproduced here verbatim.

### Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | GGUF | 0.973 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 0.961 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| qwen3.6:35b-a3b-q4_k_m | GGUF | 0.953 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 0.840 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 3 |

### Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:26b-a4b-it-q4_K_M | GGUF | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 48 | 353 | 16.0 | 225 |
| gemma4:31b-it-q4_K_M | GGUF | 4 | 4 | 4 | 4 | 3 | ✅ all Qs + bbox | 10 | 178 | 64.5 | 56 |
| qwen3.6:35b-a3b-q4_k_m | GGUF | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 57 | 525 | 14.7 | 245 |
| nemotron3:33b-q4_K_M | GGUF | 4 | 4 | 4 | 4 | 0 | ✅ all Qs + bbox | 60 | 435 | 14.6 | 247 |

## T2 — Head-to-head pivot

Rendered by `vision-suite/summarize_head_to_head.py`.

| test | metric | gemma4:26b-a4b-it-q4_K_M | gemma4:31b-it-q4_K_M | qwen3.6:35b-a3b-q4_k_m | nemotron3:33b-q4_K_M |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.973 | 0.961 | 0.953 | 0.840 |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.810 | 0.760 | 0.333 | 0.061 |
| fine text | 22/16/12/9/7 px | 4/4/4/4/3 | 4/4/4/4/3 | 4/4/4/2/2 | 4/4/4/4/0 |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 |
| throughput | gen tok/s | 48 | 10 | 57 | 60 |
| throughput | prefill tok/s | 353 | 178 | 525 | 435 |
| latency | s/req (unique image) | 16.0 | 64.5 | 14.7 | 14.6 |
| latency | req/h (serial) | 225 | 56 | 245 | 247 |

## Regression verdict vs the deployed `e74f16d8`

**No change.** Every quality cell reproduces to three decimals — scene 0.973 / 0.961 /
0.953 / 0.840, document 0.810 / 0.760 / 0.333 / 0.061 — and throughput is within noise
(48/10/57/60 vs 48/10/57/61). The 29 commits, including the breaking context-cost change
and the new scheduler fix, are **behaviour-preserving on this workload**. Determinism holds
as ADR 0012 §4 states: bit-reproducible at temperature 0 per (payload, backend, budget,
image).

## Reading the results

- **`gemma4:26b-a4b` is the best model on this host**, and by a wide margin on cost: best
  scene IoU (0.973) and best document grounding (0.810) at **225 req/h**, against 31b's
  **56** — 3.8 B active parameters versus 30.7 B dense. With fine text now tied at
  `4/4/4/4/3`, **31b has no remaining advantage here.**
- **Extraction is universal; grounding is not.** All four score 5/5 items, 5/5 qty+price,
  total and invoice, and all pass every multi-image question. The spread is entirely in
  box geometry: 0.810 / 0.760 for gemma4 against 0.333 (qwen3.6) and 0.061 (nemotron3).
  Route by task — gemma4 when boxes matter, the others are ~10 % faster for pure extraction.
- **nemotron3 falls off a cliff at 7 px** (0/4 despite 4/4 at 9 px), where both gemma4
  models degrade gracefully to 3/4.

## Method note — the fine-text assets are committed, not generated

An earlier campaign reported gemma4:26b at **3/4** on the 9 px tier. That run was missing
`visimgs/finetext.png` and `finetext_gt.json`, which are **committed assets** (`30cd94f9`),
and had regenerated a substitute. With the real assets 26b scores **4/4**, tying 31b and
removing the only cell where 31b led. `finetext_probe.py` reads those files directly and
fails with `FileNotFoundError` when absent — so a campaign that renders `—` in the OCR
columns has not measured them, and ADR 0012 §3's "`—` for unmeasured" is doing its job.
