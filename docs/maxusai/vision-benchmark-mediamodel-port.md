# Head-to-head: gemma4 media-model port vs `release/mlx-vision`

T2 per [ADR 0012](adr/0012-benchmark-report-templates.md); the table is emitted by
`vision-suite/summarize_head_to_head.py`, never typed.

- **Date:** 2026-08-11
- **Host:** Apple M5 Max, macOS 26.6, 128 GiB
- **Power mode:** `pmset -g powermode` = 0 (high power), identical for both arms
- **Server:** fork Go binary, MLX runner payload, no llama.cpp patchset (MLX path)
  - `base_mlxvision` — `release/mlx-vision` @ `98efbb7e`, version `0.32.5-maxusai-98efbb7e`
  - `port_mediamodel` — `feat/gemma4-mediamodel-port` @ `6fbf8a10`, version `0.32.5-maxusai-fix3167eb`
- **Endpoint:** `/api/generate`, `format:"json"`, `think` off
- **Sampling:** temperature 0, `num_predict` 2200, `num_ctx` 16384
- **Model:** `gemma4:12b-nvfp4`, MLX store (`OLLAMA_MODELS=~/.ollama/models-mlx`), `:11436`
- **Method:** cold server per arm, one arm at a time, no other client on the endpoint

| test | metric | base_mlxvision | port_mediamodel |
|---|---|---|---|
| scene | bbox IoU | 0.948 | 0.948 |
| scene | labels / serial | 6/6, ✅ | 6/6, ✅ |
| document | items / qty+price / total / invoice | 5/5, 5/5, ✅, ✅ | 5/5, 5/5, ✅, ✅ |
| document | name_bbox IoU | 0.638 | 0.638 |
| fine text | 22/16/12/9/7 px | 4/4/3/0/0 | 4/4/3/0/0 |
| multi (3 img) | q1 / q2 / q4-bbox / chart | ✅ ✅ ✅ 5/5 | ✅ ✅ ✅ 5/5 |
| throughput | gen tok/s | 52 | 52 |
| throughput | prefill tok/s | 1788 | 1647 |
| latency | s/req (unique image) | 11.3 | 11.5 |
| latency | req/h (serial) | 320 | 314 |

## Reading

**Quality is unchanged on every metric** — scene bbox IoU, labels, document
items/qty/price/total, name bbox IoU, the fine-text tiers, and all four
multi-image answers are identical. At temperature 0 these cells are
bit-reproducible per ADR 0012 §4, so identical here means identical, not close.

**Generation throughput is unchanged** (52 tok/s both arms).

**Prefill is ~8% slower** (1788 → 1647 tok/s), carrying ~2% into the latency pair
(11.3 → 11.5 s/req; 320 → 314 req/h). The latency deltas are within typical
run-to-run variance for a single paired run; the prefill delta is larger and has
a plausible systematic cause — the port encodes media lazily per item on first
chunk overlap (upstream's manifest) where the pre-merge path encoded every image
before prefill began. Not re-measured across repeats, so treat ~8% as an upper
bound on a real effect rather than a confirmed regression.

## Why this run exists

It found a crash the test suite did not: multi-chunk prefill with bidirectional
media produced garbage logits, surfacing as `constrained sampling produced an
illegal token` on `multi_3img`. See `fix(phase 3)` and ADR 0021's Conformance.
The numbers above are from the fixed build.
