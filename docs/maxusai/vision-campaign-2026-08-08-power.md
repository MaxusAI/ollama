# Vision campaign 2026-08-08: MLX vs GGUF across power modes

Per ADR 0012 (proposed) conventions. Provenance: Apple M5 Max 128GB, macOS 26.6,
fork `0.32.5-maxusai-0982ab8a` (:11436, cold server per model, payload b10091+001–005),
chat endpoint, think off, temp 0. Fine-text page regenerated with Courier New on this
host — absolute tier recall is not comparable to campaigns using another font.

## High Power Mode (`pmset` powermode=2, verified per model in run logs)

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox hits |
|---|---|---|---|---|---|---|
| gemma4:31b-mlx-bf16 | **MLX** | **0.966** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mlx-bf16 | **MLX** | **0.977** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-mxfp8 | **MLX** | **0.959** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-mxfp8 | **MLX** | **0.971** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-nvfp4 | **MLX** | **0.958** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-nvfp4 | **MLX** | **0.965** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:31b-it-q4_K_M | GGUF | 0.963 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 0.970 | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q4_K_M | GGUF | 0.857 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |
| nemotron3:33b-q8 | GGUF | 0.844 | 6/6 · 6/6 · 6/6 | ❌ | 5/5 · 5/5 · ✅ | 4 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-mlx-bf16 | **MLX** | 4 | 4 | 4 | 2 | 3 | ✅ all Qs + bbox | 7 | 821 | 75.1 | 48 |
| gemma4:26b-mlx-bf16 | **MLX** | 4 | 4 | 4 | 2 | 3 | ✅ all Qs + bbox | 51 | 3617 | 10.8 | 333 |
| gemma4:31b-mxfp8 | **MLX** | 4 | 4 | 4 | 1 | 3 | ✅ all Qs + bbox | 11 | 648 | 53.3 | 68 |
| gemma4:26b-mxfp8 | **MLX** | 4 | 4 | 4 | 2 | 3 | ✅ all Qs + bbox | 68 | 4009 | 8.3 | 435 |
| gemma4:31b-nvfp4 | **MLX** | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 18 | 613 | 32.6 | 111 |
| gemma4:26b-nvfp4 | **MLX** | 4 | 4 | 4 | 2 | 3 | ✅ all Qs + bbox | 85 | 4291 | 6.8 | 528 |
| gemma4:31b-it-q4_K_M | GGUF | 4 | 4 | 4 | 2 | 2 | ✅ all Qs + bbox | 17 | 301 | 36.9 | 98 |
| gemma4:26b-a4b-it-q4_K_M | GGUF | 4 | 4 | 4 | 2 | 1 | ✅ all Qs + bbox | 82 | 652 | 9.1 | 396 |
| nemotron3:33b-q4_K_M | GGUF | 4 | 4 | 2 | 1 | 0 | ✅ all Qs + bbox | 112 | 1059 | 7.1 | 508 |
| nemotron3:33b-q8 | GGUF | 4 | 4 | 3 | 0 | 0 | ✅ all Qs + bbox | 99 | 1059 | 7.7 | 469 |

Nemotron quant note: q8 vs q4_K_M moved scene IoU −0.013 (noise floor) and doc IoU
+0.003 at a 12% decode cost — the scatter-limited quality floor is the model's, not
quantization; q4_K_M remains the serving quant. Decode scaling (99 vs 112 tok/s for
2× bytes) marks the hybrid arch compute-bound, unlike bandwidth-bound dense gemma4.

## Low Power Mode

**PENDING** — awaiting host power-mode flip; identical model set and method.

## Pending additions

- nemotron3:33b-bf16 (pull in progress 2026-08-08) — both power modes.
