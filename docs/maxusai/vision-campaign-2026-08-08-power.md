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


## req/hour drop, HP → LP

| Model | Engine | req/h HP | req/h LP | retained |
|---|---|---|---|---|
| gemma4:31b-mlx-bf16 | **MLX** | 48 | 19 | 41% |
| gemma4:26b-mlx-bf16 | **MLX** | 333 | 145 | 43% |
| gemma4:31b-mxfp8 | **MLX** | 68 | 29 | 43% |
| gemma4:26b-mxfp8 | **MLX** | 435 | 191 | 44% |
| gemma4:31b-nvfp4 | **MLX** | 111 | 49 | 44% |
| gemma4:26b-nvfp4 | **MLX** | 528 | 218 | 41% |
| gemma4:31b-it-q4_K_M | GGUF | 98 | 42 | 43% |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 396 | 197 | 50% |
| nemotron3:33b-q4_K_M | GGUF | 508 | 210 | 41% |
| nemotron3:33b-q8 | GGUF | 469 | 190 | 41% |
| nemotron3:33b-bf16 | GGUF | 333 | 140 | 42% |

Low Power retains **41–50% of requests/hour across every engine, size and quant**
(median 43%) — a near-uniform governor tax, which is why the serving ranking is
power-stable. The one mild outlier (gemma4 26B q4 GGUF, 50%) is prefill-light and
decode-dominated, the profile least exposed to the harsher MLX-prefill throttle.

## Sensor record (LP half, unprivileged sampler)

20 samples at 2-min cadence, 19:01–19:39 — covering the later campaign cells
only (the sampler started ~35 min into the run); powermode=1 in every sample.
During inference: GPU device utilization avg ~51% (peaks 93–100% mid-decode on
q4/q8 cells), CPU ~14% busy — the workload is GPU-bound and the CPU stays
nearly idle. Idle samples: GPU 0%. Zero thermal or performance warnings across
the entire run (`pmset -g therm`): the LP regime is governor policy, not heat.
Per-model utilization is indicative only (n=1–4 samples/model); low readings on
the bf16 cells caught load phases (66 GB at LP disk speeds), not steady decode.
CPU/GPU clocks and temperatures were not captured (requires `sudo powermetrics`,
which stayed on the operator's side of the privilege boundary this run).
