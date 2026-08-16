# Vision campaign 2026-08-16 — Qwen3.8 first vision baseline, both engines

First vision-suite run for Qwen3.8, taken the day native support landed
(#111 … #119). Four configurations: GGUF and MLX, each think-off and think-on.

## Provenance

| | |
| --- | --- |
| server | `0.32.5-maxusai-1de352ef` (native macOS, no container) |
| store | `~/.ollama/models-mlx`, served on `:11436` |
| models | `qwen3.8:27b-q4_K_M` (GGUF → llama-server), `qwen3.8:27b-nvfp4` (MLX) |
| power | `powermode 2` (high) for the whole of every run |
| sampling | `sampling_source = packaged-defaults-no-card` — correct for this family per [ADR 0026](adr/0026-qwen38-baselines-record-the-effort-directive.md); no `CARD_THINKING` entry, and none should be added |
| think | `THINK=false` / `THINK=on` (the literal `on`, per [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md)) |
| cache | cold — daemon restarted before the pair, runs taken back to back |
| fixtures | committed `visimgs/` assets, unmodified |

## Results (T2, rendered by `summarize_head_to_head.py`)

| test | metric | qwen38-27b-q4_K_M | qwen38-27b-q4_K_M-thinkon | qwen38-27b-nvfp4 | qwen38-27b-nvfp4-thinkon |
|---|---|---|---|---|---|
| scene | bbox IoU | 0.991 (16384) | 0.980 (16384) | 0.987 (16384) | 1.000 (16384) |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.638 (16384) | 0.643 (16384) | 0.540 (16384) | 0.248 (16384) |
| fine text | 22/16/12/9/7 px | 4/4/4/2/1 (32768) | 4/4/4/2/0 (32768) | 4/4/4/2/0 (32768) | 4/4/4/1/0 (32768) |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ❌ 5/5 (16384) | ✅ ✅ ❌ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) | ✅ ✅ ✅ 5/5 (16384) |
| throughput | gen tok/s | 20 | 14 | 26 | 35 |
| throughput | prefill tok/s | 415 | 434 | 403 | 329 |
| latency | s/req (unique image) | 33.1 | 98.8 | 27.7 | 36.6 |
| latency | req/h (serial) | 109 | 36 | 130 | 98 |

## Reading it

**Think-on is a net loss and should be off for Qwen3.8 vision work.** Every
exact-match metric — labels, serial, invoice items, qty+price, totals, chart
values — is perfect in all four configurations, so thinking has nothing to
improve there. What it changes it mostly makes worse: MLX's document
`name_bbox` IoU collapses 0.540 → 0.248, MLX loses a 9px tier, GGUF loses its
7px code. The single gain is MLX's scene IoU reaching 1.000. The cost is
severe: GGUF falls to a third of its throughput, 109 → 36 req/h. That verdict
matches [ADR 0022](adr/0022-thinking-is-off-for-vision-work.md) and the
qwen3.6 verdict in ADR 0023, and is recorded here as this family's own
on-policy measurement rather than inherited.

**Quality between the engines is a tie; MLX is faster.** Think-off, the two
split the only two hard cases — MLX takes the cross-image bbox GGUF misses,
GGUF resolves one 7px code MLX does not — and MLX runs ~20% more requests per
hour.

**The `multi_3img` q4-bbox split is two different failures wearing one
symbol**, which is why `bbox_contract` now exists (`vision_suite.py`). MLX put
DYNAMO's centre within 2px of truth (348.5, 729.5 against 350, 730) but
answered normalized-1000 when the prompt asked for pixels; the scorer's
dialect tolerance rescued it into a ✅. GGUF answered in a frame of its own and
earned a ❌ indistinguishable from a units mistake. The new probe scores
grounding in the space the model declares and scores the declaration
separately.

> **Correction (2026-08-16).** This section originally said GGUF "obeyed the
> space and mislocated the shape by ~250px", and the paragraph below called it
> "genuine grounding error, not upscaling". Both are wrong. The GGUF boxes are
> truth × **1.304** uniformly: raw IoU 0.079, but **0.909** once the factor is
> divided out. The shape was always found; only the frame was wrong. The
> `bbox_contract` probe later had the model declare that frame itself —
> `ref_size [2500, 1406]` for a 1920×1080 input, the same ≈1.30×. See
> [vision-bbox-coordinate-conventions.md](vision-bbox-coordinate-conventions.md).

**The `--image-min-tokens 1024` floor is not implicated.** `prompt_eval_count`
is identical across engines on every test (2615 / 2743 / 6136 vs 6135 / 2484),
so both saw the same image geometry, and the floor only binds below ~1024
visual tokens while every fixture here is well above it — `scene_hd` alone
costs ~2042. See
[vision-token-budgets-by-arch.md](vision-token-budgets-by-arch.md) for whose
floor that is. What the floor does not explain, the model's own declared
`ref_size` does.

## Caveat on the throughput rows

**Timings on this host are not stable enough to cite as a baseline.** The same
model at the same settings measured ~11 gen tok/s in one run and ~20 in
another; a third run of MLX moved 17 → 26. Quality was byte-identical across
those repeats, as it should be at temperature 0. Treat `req/h` as the more
robust of the two throughput rows, treat both as indicative of this host on
this day, and do not compare them against numbers from another machine. The
quality rows are the citable part of this campaign.

An earlier attempt at these runs was discarded rather than reported: it began
under `powermode 0` and continued after a switch to `powermode 2`, and its
first two GGUF tests read `prefill_tps` of 34,029 and 21,339 — prompt-cache
hits warmed by an aborted run, not measurements.

## Not covered

- `cuda-dynres-005` and the ROCm lineage: not measured, and the GGUF numbers
  here would not transfer anyway.
- `bbox_contract` is not in this table. It postdates these runs; its first
  results are in the commit that adds it.
