# ROCm 10.0.0 vs 7.2.4 on gfx1151 — throughput

Measured 2026-09-20 on the Strix Halo host, `qwen35moe` suite, think-off.

**Result: ROCm 10.0.0 is not a performance reason to upgrade.** Decode gains
+0.5% to +6.4%; prefill — the image-encode path, which is what vision work is
bound by — regresses on **all five** models, −3.3% to −10.4%.

## What was held constant

Both arms carry the **identical llama.cpp payload `391fac164` (b10969)** and the
same fork build, and ran on the same host against the same checkpoints. ROCm is
the only variable:

| arm | tag prefix | ROCm | build |
|---|---|---|---|
| baseline | `gate4_0342` | 7.2.4 (`librocblas.so.5.2.70204`) | `0.34.2-dynres-f67b1aef` |
| candidate | `rocm10` | 10.0.0 (`ROCM_VERSION` stamp) | `0.34.2-dynres-rocm10probe` |

## Noise floor: ±0.4%

A few percent is unreadable without it. Two direct-I/O A/B pairs — same image,
same ROCm, a knob with no measured score effect — give the instrument's own
spread on this host, through the same generator and the same median-of-paired-
ratios statistic:

| pair | gen Δ | prefill Δ |
|---|---|---|
| `diomainoff` vs `diomainon` | +0.1 … +0.2% | −0.1 … +0.2% |
| `dio906off` vs `dio906on` | +0.0 … +0.2% | +0.1 … +0.4% |

So ±0.4%. Every bolded figure below is ten to twenty-five times that.

## `prefill_tps` had to be split first (SPEC H22)

On a KV-cache hit the server reports the whole `prompt_eval_count` against a
collapsed `prompt_eval_duration`, so the quotient jumps 5–30x while the encoder
does almost nothing. Reported unsplit, a median over mixed blocks states the
arm's cache-hit rate under the name "prefill throughput".

This is not hypothetical — it produced two wrong statements before it was
caught. `gemma4:31b` reads 168 tok/s on 0.34.1 and 1144 tok/s on 0.34.2 over
the **same 27 blocks**, and that 7x was first published as a compute win
attributed to [ADR 0036](adr/0036-gemma4-image-chunk-decodes-in-one-batch.md).
It is not compute. Cold-prefill is 167 → 172 tok/s; what changed is that 5 of
27 blocks hit cache on 0.34.1 against 20 of 27 on 0.34.2:

| arm | build | cold | cached | cold-prefill median |
|---|---|---|---|---|
| `diomainoff` | 0.34.1 | 22 | 5 | 166.7 tok/s |
| `diomainon` | 0.34.1 | 22 | 5 | 166.8 tok/s |
| `dio906off` | 0.34.1 | 22 | 5 | 165.9 tok/s |
| `gate4main` | 0.34.1 | 22 | 5 | 167.7 tok/s |
| `gate4_0342` | 0.34.2 | 7 | 20 | 172.1 tok/s |

### What moved the cache-hit rate — four causes eliminated

The change is specific to `gemma4:31b`; `gemma4:26b-a4b` was already 7/20 on
0.34.1 and did not move. Four candidates are ruled out, each by evidence rather
than by argument:

- **Not the direct-I/O knob.** Both `dio` settings, on and off, give the
  identical 22/5 split.
- **Not the harness.** `gate4main` is 0.34.1 at the same `OLLAMA_NUM_PARALLEL=2`
  as the 0.34.2 arms and still reads 22/5, so the parallel-slot count — which
  changes how KV cache is carved — is held constant across the move.
- **Not [ADR 0036](adr/0036-gemma4-image-chunk-decodes-in-one-batch.md)**, which
  was this document's first guess and is refuted by its own closing line: *"the
  floor is denied at every quantization on gfx1151, so this ADR is currently
  inert on the fork's own production hardware."* It cannot have caused a change
  on the hardware where it does not take effect.
- **Not the fork's own cache code.** `git diff 16649e8c..f67b1aef` over every
  path matching `*cache*` is MLX-runner path renames (`x/mlxrunner` →
  `mlxrunner`) and nothing else; the `runner/` delta is README deletions. Compat
  906's retirement is also excluded — the `dio906` arms carry it and read 22/5
  like the rest.

What remains is the llama.cpp payload, **b10864 → b10969** (`a43fad18`), which is
the only thing in the 84-commit window that touches the ROCm inference path.
That is 105 upstream commits; none of the server-side commits in it names prompt
reuse (the closest, `b0dcb8192` "server: fix speculation after an image", is a
drafter position fix and speculation is not enabled here). **It is localised to
that bump, not attributed within it** — bisecting 105 upstream commits at roughly
an hour per build is not proportionate for a change that is an improvement we
already have.

### What the improvement is worth

Same 27 blocks, same host, same `NUM_PARALLEL`:

| model | arm | prefill total | suite total | cached |
|---|---|---|---|---|
| `gemma4:31b` | 0.34.1 `gate4main` | 301.6 s | 1573.3 s | 5/27 |
| `gemma4:31b` | 0.34.2 `gate4_0342` | **117.0 s** | 1430.6 s | 20/27 |
| `gemma4:26b-a4b` | 0.34.1 `gate4main` | 56.3 s | 341.7 s | 20/27 |
| `gemma4:26b-a4b` | 0.34.2 `gate4_0342` | 56.1 s | 335.6 s | 20/27 |

`gemma4:31b` spends **61% less time in prefill** across the suite — 3.1 minutes
saved on a 26-minute run. End to end the suite is only 9% faster because decode
dominates at 9–10 tok/s, so this is a latency win on repeat queries against an
image already seen, not a throughput win in general. It is also worth more than
anything ROCm 10.0.0 does in either direction.

The same dilution flattened the ROCm result. Split by class, the gemma4 models
move from "flat" to the two worst regressions in the set:

| model | Δ prefill, all blocks | Δ prefill, cold only |
|---|---|---|
| `gemma4:31b` | +0.3% | **−5.5%** |
| `gemma4:26b-a4b` | −2.3% | **−10.4%** |

## Scores: 7 of 135 blocks move, and the control says that is real

The control comes first: two same-arm pairs differ in **0/54 and 0/54** blocks,
so this suite is bit-stable run-to-run on this host with think off. Against that
baseline, ROCm 10.0.0 changes **7 of 135** blocks — 5 better, 2 worse, all of
them `contract_followed` flips on known-borderline items. **Zero degenerate rows
on both arms**, so the AMD gate's clause-4 degenerate criterion is clear.

Different kernels, different rounding, borderline decisions landing the other
way. The direction is close to a coin toss and should not be read as an
improvement.

## Tables

Generated by `vision-suite/summarize_tps.py` and pasted verbatim
([ADR 0012](adr/0012-benchmark-report-templates.md) rule 8). `class` is the
block's cache classification; `n/c` marks a pair whose class differs between
arms, which is never divided.


### gemma4_26b-a4b-it-q4_K_M

| test | class | gen 7.2.4 | gen 10.0.0 | Δ gen | prefill 7.2.4 | prefill 10.0.0 | Δ prefill | prompt tok |
|---|---|---|---|---|---|---|---|---|
| bbox_contract | cache | 52.25 | 54.02 | +3.4% | 3,437.8 | 3,271.7 | -4.8% | 1511 |
| bbox_contract_adv_norm1 | cache | 47.12 | 51.07 | +8.4% | 9,293.2 | 9,113.7 | -1.9% | 3531 |
| bbox_contract_adv_real | cold | 47.29 | 50.42 | +6.6% | 412.2 | 369.9 | -10.3% | 3549 |
| bbox_contract_anchored | cache | 47.32 | 51.20 | +8.2% | 9,232.5 | 9,008.2 | -2.4% | 3654 |
| bbox_contract_anchored_1img | cold | 52.10 | 54.28 | +4.2% | 327.0 | 292.7 | -10.5% | 1497 |
| bbox_contract_box2d_1img | cache | 52.35 | 54.36 | +3.8% | 4,365.7 | 4,219.2 | -3.4% | 1500 |
| bbox_contract_multi | cold | 47.08 | 51.15 | +8.6% | 286.1 | 256.1 | -10.5% | 3689 |
| bbox_contract_perobject | cache | 47.48 | 51.14 | +7.7% | 11,713.5 | 11,379.0 | -2.9% | 3551 |
| bbox_contract_pinned | cache | 47.47 | 50.94 | +7.3% | 8,334.3 | 8,237.0 | -1.2% | 3538 |
| bbox_contract_positional_1img | cache | 52.36 | 54.53 | +4.1% | 4,271.7 | 4,324.4 | +1.2% | 1520 |
| bbox_contract_real_1img | cache | 52.20 | 54.40 | +4.2% | 3,113.3 | 3,083.2 | -1.0% | 1457 |
| bbox_contract_reasoning | cache | 47.23 | 50.99 | +8.0% | 7,441.4 | 7,172.3 | -3.6% | 3676 |
| bboxm_free_anc_named | cache | 52.20 | 54.41 | +4.2% | 3,334.6 | 3,271.8 | -1.9% | 1484 |
| bboxm_free_anc_pos | cache | 52.20 | 54.10 | +3.6% | 3,823.1 | 3,782.9 | -1.0% | 1535 |
| bboxm_free_noanc_named | cache | 52.46 | 54.34 | +3.6% | 4,118.7 | 4,222.9 | +2.5% | 1393 |
| bboxm_free_noanc_pos | cache | 52.43 | 54.19 | +3.4% | 3,629.9 | 3,542.9 | -2.4% | 1439 |
| bboxm_pin_anc_pos | cache | 52.15 | 54.51 | +4.5% | 3,909.0 | 3,850.3 | -1.5% | 1513 |
| bboxm_pin_noanc_named | cache | 52.36 | 54.79 | +4.6% | 3,985.4 | 4,082.9 | +2.4% | 1378 |
| bboxm_pin_noanc_pos | cache | 52.26 | 54.48 | +4.2% | 3,838.5 | 3,752.7 | -2.2% | 1424 |
| document_single | cold | 52.37 | 54.41 | +3.9% | 323.4 | 289.8 | -10.4% | 1447 |
| finetext | cold | 52.92 | 55.51 | +4.9% | 273.4 | 243.3 | -11.0% | 1173 |
| multi_3img | cold | 46.92 | 50.65 | +7.9% | 428.1 | 385.0 | -10.1% | 3762 |
| multi_3img_anchored | cache | 47.05 | 50.67 | +7.7% | 13,279.3 | 12,998.8 | -2.1% | 3846 |
| scene_single | cache | 51.77 | 53.93 | +4.2% | 3,848.0 | 3,870.5 | +0.6% | 1684 |
| scene_single_anchored | cache | 51.39 | 53.47 | +4.0% | 5,713.5 | 5,362.1 | -6.1% | 1822 |
| scene_single_pinned | cache | 51.63 | 53.66 | +3.9% | 2,911.0 | 2,874.7 | -1.2% | 1736 |

1 block(s) dropped as `cold_start` (first request after restart).

### gemma4_31b-it-q4_K_M

| test | class | gen 7.2.4 | gen 10.0.0 | Δ gen | prefill 7.2.4 | prefill 10.0.0 | Δ prefill | prompt tok |
|---|---|---|---|---|---|---|---|---|
| bbox_contract | cache | 9.87 | 10.36 | +5.0% | 861.8 | 866.9 | +0.6% | 1511 |
| bbox_contract_adv_norm1 | cache | 8.99 | 9.70 | +7.9% | 2,765.6 | 2,833.3 | +2.4% | 3531 |
| bbox_contract_adv_real | cold | 8.95 | 9.69 | +8.3% | 221.0 | 210.3 | -4.8% | 3549 |
| bbox_contract_anchored | cache | 8.97 | 9.65 | +7.6% | 2,767.1 | 2,827.6 | +2.2% | 3654 |
| bbox_contract_anchored_1img | cold | 9.66 | 10.31 | +6.7% | 164.2 | 161.5 | -1.6% | 1497 |
| bbox_contract_box2d_1img | cache | 9.66 | 10.31 | +6.7% | 1,337.2 | 1,358.4 | +1.6% | 1500 |
| bbox_contract_multi | cold | 8.98 | 9.68 | +7.8% | 156.6 | 147.4 | -5.9% | 3689 |
| bbox_contract_perobject | cache | 9.00 | 9.69 | +7.7% | 4,168.8 | 4,243.5 | +1.8% | 3551 |
| bbox_contract_pinned | cache | 9.03 | 9.71 | +7.5% | 2,290.2 | 2,345.9 | +2.4% | 3538 |
| bbox_contract_positional_1img | cache | 9.62 | 10.31 | +7.2% | 1,354.6 | 1,384.3 | +2.2% | 1520 |
| bbox_contract_real_1img | cache | 9.72 | 10.33 | +6.3% | 839.6 | 912.4 | +8.7% | 1457 |
| bbox_contract_reasoning | cache | 9.00 | 9.70 | +7.8% | 1,835.3 | 1,854.8 | +1.1% | 3676 |
| bboxm_free_anc_named | cache | 9.72 | 10.31 | +6.1% | 958.3 | 957.5 | -0.1% | 1484 |
| bboxm_free_anc_pos | cache | 9.66 | 10.28 | +6.4% | 1,085.3 | 1,082.2 | -0.3% | 1535 |
| bboxm_free_noanc_named | cache | 9.86 | 10.40 | +5.5% | 1,347.0 | 1,368.6 | +1.6% | 1393 |
| bboxm_free_noanc_pos | cache | 9.81 | 10.37 | +5.7% | 1,185.0 | 1,182.6 | -0.2% | 1439 |
| bboxm_pin_anc_pos | cache | 9.69 | 10.31 | +6.4% | 1,080.2 | 1,078.1 | -0.2% | 1513 |
| bboxm_pin_noanc_named | cache | 9.88 | 10.39 | +5.2% | 1,322.2 | 1,353.6 | +2.4% | 1378 |
| bboxm_pin_noanc_pos | cache | 9.83 | 10.37 | +5.5% | 1,260.6 | 1,265.0 | +0.3% | 1424 |
| document_single | cold | 9.70 | 10.31 | +6.3% | 172.1 | 163.3 | -5.1% | 1447 |
| finetext | cold | 10.22 | 10.51 | +2.8% | 157.0 | 146.7 | -6.6% | 1173 |
| multi_3img | cold | 8.90 | 9.59 | +7.8% | 221.5 | 207.9 | -6.1% | 3762 |
| multi_3img_anchored | cache | 8.90 | 9.58 | +7.6% | 4,711.3 | 4,776.8 | +1.4% | 3846 |
| scene_single | cache | 9.60 | 10.21 | +6.4% | 1,143.6 | 1,136.2 | -0.6% | 1684 |
| scene_single_anchored | cache | 9.52 | 10.13 | +6.4% | 2,276.1 | 2,279.8 | +0.2% | 1822 |
| scene_single_pinned | cache | 9.58 | 10.20 | +6.5% | 733.7 | 733.3 | -0.1% | 1736 |

1 block(s) dropped as `cold_start` (first request after restart).

### nemotron3_33b-q4_K_M

| test | class | gen 7.2.4 | gen 10.0.0 | Δ gen | prefill 7.2.4 | prefill 10.0.0 | Δ prefill | prompt tok |
|---|---|---|---|---|---|---|---|---|
| bbox_contract | cold | 64.57 | 64.60 | +0.0% | 458.3 | 418.2 | -8.8% | 2493 |
| bbox_contract_adv_norm1 | cold | 63.06 | 63.53 | +0.7% | 418.6 | 382.9 | -8.5% | 5966 |
| bbox_contract_adv_real | cold | 63.11 | 63.54 | +0.7% | 420.7 | 384.1 | -8.7% | 5986 |
| bbox_contract_anchored | cold | 63.16 | 63.45 | +0.5% | 426.6 | 389.6 | -8.7% | 6097 |
| bbox_contract_anchored_1img | cold | 64.54 | 64.74 | +0.3% | 458.5 | 419.3 | -8.6% | 2492 |
| bbox_contract_box2d_1img | cold | 64.50 | 64.80 | +0.5% | 459.4 | 420.8 | -8.4% | 2493 |
| bbox_contract_multi | cold | 63.23 | 63.30 | +0.1% | 427.6 | 389.9 | -8.8% | 6120 |
| bbox_contract_perobject | cold | 63.16 | 63.43 | +0.4% | 419.7 | 383.3 | -8.7% | 5973 |
| bbox_contract_pinned | cold | 63.38 | 63.71 | +0.5% | 418.4 | 382.0 | -8.7% | 5958 |
| bbox_contract_positional_1img | cold | 64.46 | 64.29 | -0.3% | 461.3 | 420.2 | -8.9% | 2514 |
| bbox_contract_real_1img | cold | 64.38 | 64.71 | +0.5% | 453.7 | 414.4 | -8.6% | 2458 |
| bbox_contract_reasoning | cold | 63.17 | 63.47 | +0.5% | 426.6 | 389.2 | -8.8% | 6110 |
| bboxm_free_anc_named | cold | 64.39 | 64.45 | +0.1% | 458.6 | 417.9 | -8.9% | 2489 |
| bboxm_free_anc_pos | cold | 64.35 | 64.68 | +0.5% | 465.8 | 423.9 | -9.0% | 2540 |
| bboxm_free_noanc_named | cold | 64.59 | 64.49 | -0.2% | 443.8 | 403.6 | -9.1% | 2381 |
| bboxm_free_noanc_pos | cold | 63.97 | 64.41 | +0.7% | 448.3 | 407.3 | -9.1% | 2428 |
| bboxm_pin_anc_pos | cold | 64.50 | 64.60 | +0.2% | 458.9 | 417.8 | -9.0% | 2505 |
| bboxm_pin_noanc_named | cold | 64.46 | 64.76 | +0.5% | 439.0 | 399.2 | -9.1% | 2355 |
| bboxm_pin_noanc_pos | cold | 64.68 | 64.68 | +0.0% | 444.7 | 403.9 | -9.2% | 2402 |
| document_single | cold | 64.32 | 64.57 | +0.4% | 404.1 | 366.3 | -9.3% | 2798 |
| finetext | cold | 64.35 | 64.84 | +0.8% | 369.9 | 337.4 | -8.8% | 2495 |
| multi_3img | cold | 62.91 | 63.31 | +0.6% | 432.9 | 395.8 | -8.6% | 6202 |
| multi_3img_anchored | cold | 63.01 | 63.30 | +0.5% | 437.8 | 399.4 | -8.8% | 6294 |
| scene_single | cold | 64.38 | 64.76 | +0.6% | 484.9 | 442.1 | -8.8% | 2673 |
| scene_single_anchored | cold | 64.31 | 64.69 | +0.6% | 507.4 | 463.8 | -8.6% | 2825 |
| scene_single_pinned | cold | 64.43 | 64.60 | +0.3% | 492.4 | 447.7 | -9.1% | 2733 |

1 block(s) dropped as `cold_start` (first request after restart).

### qwen3_6_35b-a3b-q4_k_m

| test | class | gen 7.2.4 | gen 10.0.0 | Δ gen | prefill 7.2.4 | prefill 10.0.0 | Δ prefill | prompt tok |
|---|---|---|---|---|---|---|---|---|
| bbox_contract | cold | 57.93 | 61.38 | +6.0% | 549.1 | 515.9 | -6.1% | 2445 |
| bbox_contract_adv_norm1 | cold | 56.41 | 59.57 | +5.6% | 498.9 | 467.7 | -6.3% | 5915 |
| bbox_contract_adv_real | cold | 56.31 | 59.71 | +6.0% | 501.5 | 470.2 | -6.2% | 5932 |
| bbox_contract_anchored | cold | 56.24 | 59.84 | +6.4% | 501.1 | 474.5 | -5.3% | 6036 |
| bbox_contract_anchored_1img | cold | 59.07 | 61.38 | +3.9% | 543.6 | 509.4 | -6.3% | 2431 |
| bbox_contract_box2d_1img | cold | 59.78 | 61.16 | +2.3% | 544.7 | 510.8 | -6.2% | 2431 |
| bbox_contract_multi | cold | 40.90 | 59.90 | +46.5% | 505.4 | 478.2 | -5.4% | 6071 |
| bbox_contract_perobject | cold | 54.81 | 59.40 | +8.4% | 354.7 | 471.6 | +33.0% | 5932 |
| bbox_contract_pinned | cold | 41.23 | 59.60 | +44.6% | 312.9 | 471.1 | +50.5% | 5921 |
| bbox_contract_positional_1img | cold | 59.61 | 61.30 | +2.8% | 551.9 | 516.4 | -6.4% | 2451 |
| bbox_contract_real_1img | cold | 59.54 | 61.41 | +3.1% | 535.2 | 503.0 | -6.0% | 2388 |
| bbox_contract_reasoning | cold | 40.26 | 59.69 | +48.3% | 321.9 | 477.8 | +48.4% | 6059 |
| bboxm_free_anc_named | cold | 55.37 | 60.99 | +10.1% | 544.6 | 514.6 | -5.5% | 2416 |
| bboxm_free_anc_pos | cold | 59.53 | 60.86 | +2.2% | 561.8 | 526.2 | -6.3% | 2465 |
| bboxm_free_noanc_named | cold | 60.03 | 61.16 | +1.9% | 544.2 | 507.1 | -6.8% | 2328 |
| bboxm_free_noanc_pos | cold | 59.80 | 61.30 | +2.5% | 544.0 | 510.9 | -6.1% | 2373 |
| bboxm_pin_anc_pos | cold | 58.55 | 60.70 | +3.7% | 563.9 | 524.6 | -7.0% | 2444 |
| bboxm_pin_noanc_named | cold | 59.18 | 61.70 | +4.3% | 539.8 | 508.1 | -5.9% | 2313 |
| bboxm_pin_noanc_pos | cold | 58.83 | 61.71 | +4.9% | 541.3 | 511.0 | -5.6% | 2358 |
| document_single | cold | 57.23 | 61.16 | +6.9% | 482.2 | 452.7 | -6.1% | 2743 |
| finetext | cold | 60.25 | 61.83 | +2.6% | 454.4 | 422.4 | -7.0% | 2484 |
| multi_3img | cold | 55.10 | 59.14 | +7.3% | 506.3 | 482.3 | -4.7% | 6136 |
| multi_3img_anchored | cold | 55.92 | 59.60 | +6.6% | 510.3 | 484.4 | -5.1% | 6218 |
| scene_single | cold | 59.23 | 60.89 | +2.8% | 572.1 | 538.2 | -5.9% | 2615 |
| scene_single_anchored | cold | 58.17 | 60.64 | +4.2% | 591.0 | 554.5 | -6.2% | 2756 |
| scene_single_pinned | cold | 59.03 | 60.67 | +2.8% | 583.1 | 546.4 | -6.3% | 2670 |

1 block(s) dropped as `cold_start` (first request after restart).

### qwen3_8_27b-q4_K_M

| test | class | gen 7.2.4 | gen 10.0.0 | Δ gen | prefill 7.2.4 | prefill 10.0.0 | Δ prefill | prompt tok |
|---|---|---|---|---|---|---|---|---|
| bbox_contract | cold | 12.28 | 12.45 | +1.4% | 264.0 | 255.0 | -3.4% | 2445 |
| bbox_contract_adv_norm1 | cold | 12.03 | 12.31 | +2.3% | 252.1 | 243.7 | -3.3% | 5915 |
| bbox_contract_adv_real | cold | 12.06 | 12.28 | +1.8% | 252.1 | 244.7 | -3.0% | 5932 |
| bbox_contract_anchored | cold | 12.02 | 12.29 | +2.2% | 255.1 | 246.6 | -3.4% | 6036 |
| bbox_contract_anchored_1img | cold | 12.27 | 12.42 | +1.2% | 266.1 | 256.9 | -3.5% | 2431 |
| bbox_contract_box2d_1img | cold | 12.27 | 12.41 | +1.1% | 265.1 | 255.7 | -3.6% | 2431 |
| bbox_contract_multi | cold | 12.04 | 12.30 | +2.2% | 253.9 | 246.2 | -3.0% | 6071 |
| bbox_contract_perobject | cold | 12.03 | 12.32 | +2.4% | 253.3 | 244.5 | -3.5% | 5932 |
| bbox_contract_pinned | cold | 12.06 | 12.33 | +2.2% | 251.9 | 244.3 | -3.0% | 5921 |
| bbox_contract_positional_1img | cold | 12.27 | 12.44 | +1.4% | 263.6 | 255.0 | -3.3% | 2451 |
| bbox_contract_real_1img | cold | 12.29 | 12.39 | +0.8% | 261.7 | 252.3 | -3.6% | 2388 |
| bbox_contract_reasoning | cold | 12.04 | 12.32 | +2.3% | 253.1 | 245.7 | -2.9% | 6059 |
| bboxm_free_anc_named | cold | 12.04 | 12.45 | +3.4% | 256.0 | 255.0 | -0.4% | 2416 |
| bboxm_free_anc_pos | cold | 12.25 | 12.43 | +1.5% | 264.1 | 256.1 | -3.0% | 2465 |
| bboxm_free_noanc_named | cold | 12.26 | 12.43 | +1.4% | 260.1 | 252.0 | -3.1% | 2328 |
| bboxm_free_noanc_pos | cold | 12.27 | 12.46 | +1.5% | 262.8 | 253.4 | -3.6% | 2373 |
| bboxm_pin_anc_pos | cold | 12.21 | 12.43 | +1.8% | 263.4 | 254.5 | -3.4% | 2444 |
| bboxm_pin_noanc_named | cold | 12.26 | 12.48 | +1.8% | 257.9 | 250.6 | -2.8% | 2313 |
| bboxm_pin_noanc_pos | cold | 12.27 | 12.43 | +1.3% | 261.5 | 252.6 | -3.4% | 2358 |
| document_single | cold | 12.23 | 12.42 | +1.6% | 252.3 | 241.7 | -4.2% | 2743 |
| finetext | cold | 12.27 | 12.45 | +1.5% | 244.2 | 234.1 | -4.1% | 2484 |
| multi_3img | cold | 11.98 | 12.28 | +2.5% | 255.0 | 247.6 | -2.9% | 6136 |
| multi_3img_anchored | cold | 12.00 | 12.30 | +2.5% | 255.3 | 247.2 | -3.2% | 6218 |
| scene_single | cold | 12.21 | 12.41 | +1.6% | 267.1 | 258.6 | -3.2% | 2615 |
| scene_single_anchored | cold | 12.23 | 12.40 | +1.4% | 271.4 | 262.7 | -3.2% | 2756 |
| scene_single_pinned | cold | 12.23 | 12.42 | +1.6% | 271.4 | 262.4 | -3.3% | 2670 |

1 block(s) dropped as `cold_start` (first request after restart).

### Summary — median of per-test paired ratios

| model | gen 7.2.4 | gen 10.0.0 | Δ gen | n | prefill 7.2.4 (cold) | prefill 10.0.0 (cold) | Δ prefill cold | n | Δ prefill cache | n |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma4_26b-a4b-it-q4_K_M | 52.12 | 54.06 | **+4.2%** | 26 | 325.2 | 291.2 | **-10.4%** | 6 | -1.9% | 20 |
| gemma4_31b-it-q4_K_M | 9.64 | 10.29 | **+6.4%** | 26 | 168.1 | 162.4 | **-5.5%** | 6 | +1.2% | 20 |
| nemotron3_33b-q4_K_M | 64.35 | 64.53 | **+0.5%** | 26 | 444.2 | 403.7 | **-8.8%** | 26 | — | 0 |
| qwen3_6_35b-a3b-q4_k_m | 58.36 | 60.88 | **+4.6%** | 26 | 540.6 | 507.6 | **-6.1%** | 26 | — | 0 |
| qwen3_8_27b-q4_K_M | 12.23 | 12.41 | **+1.6%** | 26 | 259.0 | 252.1 | **-3.3%** | 26 | — | 0 |

Cache class agreed on every block in every model, so no block was excluded from the paired prefill delta.

host: http://127.0.0.1:11499 · builds: ['0.34.2-dynres-f67b1aef', '0.34.2-dynres-rocm10probe']

## Reproducing

```
cd docs/maxusai/vision-suite
python3 summarize_tps.py --a gate4_0342 --b rocm10 --labels "7.2.4,10.0.0"
```

Raw paired blocks: `vision-suite/bench-runs/rocm-10-vs-724-tps-2026-09-20.json`.

## Standing test instance

ROCm 10.0.0 was declined for production and is carried as an **experimental**
surface — [ADR 0040](adr/0040-rocm-10-is-experimental-until-it-is-faster.md),
with the promotion it was weighed against in
[amd-upgrade-gate.md](amd-upgrade-gate.md) (the 2026-09-21 decision). It is kept
running for further testing:

| | |
|---|---|
| container | `ollama-rocm10` |
| port | **11500** |
| image | `maxusai-ollama:rocm10-gfx1151-probe` (`0.34.2-dynres-rocm10probe`) |

**Not 11499.** That is the vision-suite scratch port, and a long-lived container
squatting on it would make the next `bench_up_np.sh` fail to bind.

It shares `/opt/ollama/.ollama` with production, which makes two settings
non-optional rather than tuning:

- `OLLAMA_NOPRUNE=1` — this instance must never prune blobs the serving
  instance depends on.
- `OLLAMA_MAX_LOADED_MODELS=1` — one model at a time. This is an iGPU sharing
  system RAM with the instance that is actually serving requests.

`OLLAMA_KEEP_ALIVE=60s` so it releases the GPU shortly after a test instead of
holding ~20 GB indefinitely; raise it if reload churn gets in the way. Docker log
caps are kept because `OLLAMA_DEBUG=1` on a long-lived container writes a lot and
docker's partition runs at 95% use.

The image is a **single-arch (gfx1151) probe build from an unmerged branch**, so
it is not reproducible from `main`. That is fine for testing and is exactly why
it was not promoted.

## Open

- ROCm 10.0.0 has only ever been run on this one host and this one suite. It
  executes correctly on gfx1151 and is numerically sane; nothing here says it is
  ready to carry production.
- The prefill regression is flat across prompt sizes within a model
  (`qwen3.6` −6.2% on small prompts vs −5.4% on large) and varies by model, so
  it tracks architecture rather than image size — consistent with kernel
  selection, not measured to it.
- `qwen3.6`'s 7.2.4 arm has four disturbed blocks (`bbox_contract_multi`,
  `_perobject`, `_pinned`, `_reasoning`) reading 40–41 gen tok/s against ~58
  elsewhere. The median statistic absorbs them; they are flagged rather than
  explained, and a repeat of that arm would settle it.
