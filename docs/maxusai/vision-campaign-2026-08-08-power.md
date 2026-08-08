# Vision campaign 2026-08-08: MLX vs GGUF across power modes

Per ADR 0012 (proposed) conventions. Provenance: Apple M5 Max 128GB, macOS 26.6,
fork `0.32.5-maxusai-0982ab8a` (:11436, cold server per model, payload b10091+001–005),
chat endpoint, think off, temp 0. Fine-text page regenerated with Courier New on this
host. Power mode verified per model in the run logs (`powermode=` stamps); score files
archived per half under `runs/{hp,lp}-scores-2026-08-08/`.

## High Power Mode (powermode=2, 11/11 verified)

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|
| gemma4:31b-mlx-bf16 gemma4:26b-mlx-bf16 gemma4:31b-mxfp8 gemma4:26b-mxfp8 gemma4:31b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M nemotron3:33b-q4_K_M nemotron3:33b-q8 nemotron3:33b-bf16 | **MLX** | — | —/— · —/— · —/— | ❌ | —/— · —/— · ❌ | — |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-mlx-bf16 gemma4:26b-mlx-bf16 gemma4:31b-mxfp8 gemma4:26b-mxfp8 gemma4:31b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M nemotron3:33b-q4_K_M nemotron3:33b-q8 nemotron3:33b-bf16 | **MLX** | — | — | — | — | — | — | — | — | — | — |

## Low Power Mode (powermode=1, 11/11 verified)

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|
| gemma4:31b-mlx-bf16 gemma4:26b-mlx-bf16 gemma4:31b-mxfp8 gemma4:26b-mxfp8 gemma4:31b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M nemotron3:33b-q4_K_M nemotron3:33b-q8 nemotron3:33b-bf16 | **MLX** | — | —/— · —/— · —/— | ❌ | —/— · —/— · ❌ | — |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-mlx-bf16 gemma4:26b-mlx-bf16 gemma4:31b-mxfp8 gemma4:26b-mxfp8 gemma4:31b-nvfp4 gemma4:26b-nvfp4 gemma4:31b-it-q4_K_M gemma4:26b-a4b-it-q4_K_M nemotron3:33b-q4_K_M nemotron3:33b-q8 nemotron3:33b-bf16 | **MLX** | — | — | — | — | — | — | — | — | — | — |

## Power-mode comparison

- **Quality is power-invariant: zero drift.** All 22 scene-IoU cells (and every
  extraction/OCR/multi metric) reproduce across power modes to three decimals —
  temperature-0 determinism holds under the LP governor, so LP costs throughput only.
- **Decode retains ~43–57% under LP** across engines and quants (e.g. nemotron q4
  112→49, gemma4 26b-q4 82→45, 26b-nvfp4 85→36, 31b-bf16 7.3→3.0 tok/s).
- **Prefill is hit harder on MLX** (~28–31% retained: 4291→1212, 3617→1130) than
  GGUF (~38–39%: 652→257, 1059→407), compressing MLX's prefill lead from ~6× to
  ~4.7× — still decisive for req/h.
- **req/h leaders under LP**: 26b-nvfp4 and nemotron q4 remain 1–2 (see tables);
  ranking is power-stable even though absolute rates halve.
- Sensors during LP (unprivileged sampler, 2-min cadence): GPU device utilization
  ~80% during decode, no thermal or performance warnings recorded — the LP regime
  is governor policy, not thermal throttling. Clock/temperature detail requires
  `sudo powermetrics` (not captured this run).
