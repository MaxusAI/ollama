# ADR 0037: keep the MLX #3912 kernel fix — decided on correctness, with OCR accuracy measured neutral

- **Status:** accepted 2026-09-19 (the maintainer: "keep 0.34.x and promote to production").
  Sits beside [ADR 0036](0036-gemma4-image-chunk-decodes-in-one-batch.md), which made the
  same trade on the GGUF path for the same cell.
- **Date:** 2026-09-19
- **Deciders:** MaxusAI fork maintainers

## Context

The v0.34.1 fold moved the MLX pin to `d9add9d1`, which carries upstream
[ml-explore/mlx#3912](https://github.com/ml-explore/mlx/pull/3912): Metal's `fp_qmm_t` ran its K
loop past `K_eff` when **K mod 32 == 16**, reading 16 columns of the next row's packed weights and
scales into the accumulator. The gemma4 vision tower's `mlp.down_proj` is nvfp4 `[1152 × 4304]`,
K = 134 × 32 + 16, in the 26b and 31b checkpoints (27 layers each). 12b has no `vision_tower.encoder`
at all, only a `vision_embedder`, and never met the defect.

On Metal the fold moved one scored cell: `gemma4:31b-nvfp4`'s 9 px fine-text tier, 4 → 3. That raised
the question this ADR answers — **should the fork revert the pin, or carry a patch restoring the old
kernel behaviour, to get the tier back?**

The evidence, all measured on Apple Silicon `10.8.0.3` against the deployed build
`0.34.0-maxusai-8a7ba949` and the archived `0.33.2-maxusai-2b95b4a5`:

**The fixed kernel is arithmetically correct and the old one was not.** The guard test
(`mlx/fp_qmm_t_kmod32_test.go`, #315) compares `QuantizedMatmul` against
dequantize-then-matmul at M = 256, N = 1152, group 16:

```
MLX ce916dbb (pre-fix)   K=4288  max 0.000152588   0/294912 elements off by > 1.0
                         K=4304  max 24.0494       232722/294912
MLX d9add9d1 (post-fix)  K=4288  max 0.000152588   0/294912
                         K=4304  max 0.137939      0/294912
```

`TestVisionGoldenParity`'s max sampled element delta — the fused quantized matmul against
dequantize-then-matmul on real weights — improves `0.1406 → 0.1094` on 31b across the fold and to
`0.0898` with the separate global-scale round trip removed, **exactly** MLX-CUDA's `0.0898`, where the
defective kernel never existed ([upstream-sync-0.34.1.md](../tasks/upstream-sync-0.34.1.md)
§ "The MLX pin move and the vision encoder", #316).

**The 9 px tier is not a readout of encoder correctness.** Ten reps per quantization, think-off, one
binary and window (`bench-runs/finetext-9px-31b-quant-reps-2026-09-18.json`):

| checkpoint | vision `down_proj` | rest of tower | LM | 9 px at 4 |
|---|---|---|---|---|
| `31b-mlx-bf16` | bf16 | bf16 | bf16 | **10/10** |
| `31b-mxfp8` | bf16 | 8-bit | 8-bit | **0/10** |
| `31b-nvfp4` | 4-bit | 4-bit | 4-bit | **2/10** |

`31b-mxfp8` carries a bf16 `down_proj` — the exact tensor #3912 corrupts — and still scores below the
all-4-bit checkpoint. The cell does not order by numerical precision, and it is bimodal: across both
of the suite's finetext arms it read 4 on 14 of 14 pre-fix captures and 1 of 16 post-fix, with one
0.34.0 pair of captures disagreeing at byte-identical provenance by a single character (`RNK-0391-DW18`
against `RMK-0391-DW18`, ground truth `RNK`).

It is also the cell ADR 0036 already moved. On the GGUF path, decoding a gemma4 image chunk in one
batch — the bidirectionally faithful path — takes 31b's 9 px tier 4 → 3 deterministically, and the
fork accepted that as the cost of the faithful computation. **Two unrelated correctness fixes, on two
engines, move this cell the same way.**

**On real OCR the defect is undetectable.** All 1000 OCRBench v1 items, the same local checkpoint
under both binaries (`bench-runs/ocrbench-v1-1000-gemma4-31b-nvfp4-0340-vs-0332.json`):

```
0.34.0-maxusai-8a7ba949 (MLX d9add9d1, fixed)    835/1000 = 0.8350
0.33.2-maxusai-2b95b4a5 (MLX c793734e, defect)   833/1000 = 0.8330

both right 826 | both wrong 158 | fixed only 9 | defect only 7
discordant 16   exact McNemar two-tailed p = 0.8036
```

## Decision

1. **Keep the MLX pin at `d9add9d1`, with #3912.** No revert of the pin, and no compat patch
   reintroducing the pre-fix `fp_qmm_t` loop bound.
2. **The decision rests on correctness, not accuracy.** There is no accuracy argument on either side —
   OCRBench cannot tell the two kernels apart — and there is a correctness argument on one. A fork that
   carries a known-wrong kernel because one glyph scored higher would be optimising the benchmark,
   not the model.
3. **A fine-text size tier does not gate a pin, kernel or quantization decision on its own.** A tier
   move across a build is a prompt to measure (the unquantized arm on the same binary and window, a
   rate rather than one capture, and a many-item external benchmark), not a verdict. The measurement
   rules are [SPEC vision-harness-reuse](../spec/vision-harness-reuse.md) H15–H18.
4. **The mlx-metal surface is promoted on this fold.** `0.34.0-maxusai-8a7ba949` serves `:11435`; its
   preflight run (`preflight/runs/preflight-mlx-metal-0340-8a7ba949.json`, profile `mlx-metal-0-34-0`,
   PASS 19 / SKIP 12) is recorded, and the README's release matrix and deploy line carry it.

## Options considered

- **Revert the MLX pin to the pre-fix commit.** Restores the tier on one checkpoint. Rejected: it
  reinstates a kernel wrong on 79 % of that matmul's outputs at the served shape, for a cell that
  OCRBench shows carries no accuracy, and it drops every other fix in the pin range.
- **Carry a compat patch restoring the old loop bound.** Same outcome as reverting with more
  maintenance, and a patch whose whole purpose is to reintroduce an upstream-fixed memory over-read.
- **Keep the fix but gate the fold on the 9 px tier until it returns to 4.** Rejected: the tier does not
  order by precision (the bf16-`down_proj` checkpoint scores 0/10), and ADR 0036 already accepted the
  same move on the other engine.
- **Keep the fix (chosen).**

## Consequences

- **Every Metal 26b/31b vision number taken before `8a7ba949` went through the defective kernel** and is
  superseded by [the 2026-09-18 campaign](../vision-campaign-2026-09-18-mlx8a7ba949-nvfp4.md).
- **The registry has drifted from the checkpoints measured here.** Per the MLX-CUDA session's registry
  audit (#312), the library's current `gemma4:31b-nvfp4` ships a bf16 vision tower where the checkpoint
  measured here has a 4-bit one — 194 layers differ. A bf16-tower checkpoint never meets #3912 on any
  build. The two share a config digest, which is expected and means nothing: the config digest is the
  architecture's `config.json`, shared even by our local nvfp4 and bf16 checkpoints. The measured
  checkpoints are identified by manifest digest (`31b-nvfp4` `637cc0ff…`, `31b-mxfp8` `1434769c…`,
  `31b-mlx-bf16` `fb3f25b3…`) and preserved under tower-qualified tags
  (`gemma4:31b-nvfp4-tower-nvfp4`, `-mxfp8-tower-mxfp8`, `-mlx-bf16-tower-bf16`), whose manifest digests
  match the originals, so a pull cannot silently replace what was measured.
- **The build stamps disagreed with ADR 0032 — resolved 2026-09-19.** The Metal build stamped
  `0.34.0-maxusai-<sha>` where the CUDA image stamps `0.34.1-dynres-0-g<sha>` for the same commit, so
  one `--version` could not render both matrix rows. `build-macos.sh` now stamps through
  `scripts/env.sh`, `mlx-metal-0-34-0` admits both stamps of the deployed build and refuses a fold build
  made before its tag, `release_matrix.py --version` is repeatable, and the equivalence
  `0.34.0-maxusai-8a7ba949` ≡ `0.34.1-dynres-0-g8a7ba94` is recorded
  ([ADR 0032](0032-fork-version-identity-tags-each-upstream-fold.md), 2026-09-19 amendment).
- **The Metal deploy lacked `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR=0` — resolved 2026-09-19.** The CUDA
  container carries it (ADR 0033; the fold's finding that drafting under a grammar retains memory without
  bound). It was added to the launchd environment and the service re-bootstrapped at 17:44:40; the running
  server's own environment shows it.
- **Parked 2026-09-19:** the nvfp4 global-scale representation (raw `m` vs `× 2688 / 2688`), and with it
  #287 (the maintainer: "park #287 for now"). Independent of this decision — the round trip moves the encoder by
  one f32 ulp on 17 of 31b's 191 vision scales and provably does not move the 9 px tier. A proposed ADR on
  exactly this question, "nvfp4 global scales are stored as the checkpoint multiplier", sits in #323 and is
  parked with it.
- **Retirement:** none. This is upstream's fix, carried by the pin, not a fork change.
