# ADR 0039: nvfp4 global scales are stored as the checkpoint multiplier

- **Status:** proposed 2026-09-19. Awaiting Glenn. Arises from
  [#312](https://github.com/MaxusAI/ollama/issues/312) and gates part of
  [#287](https://github.com/MaxusAI/ollama/pull/287).
- **Date:** 2026-09-19
- **Deciders:** MaxusAI fork maintainers

## Context

An nvfp4 checkpoint carries a per-tensor multiplier `m` (ModelOpt's `weight_scale_2`).
MLX wants that scale in its own representation, `m × 448 × 6` — `Nvfp4MaxProduct`, 2688 —
because its kernels fold the maxima of the E4M3 and E2M1 ranges into the scale.

The v0.34 fold made that representation the **stored** form.
`ToMLXGlobalScale` (`mlxrunner/model/quant.go`) converts once at load:

```go
return mlx.MulScalar(flat, mlx.Nvfp4MaxProduct)   // holds f32(m × 2688)
```

and every wrapper that applies the scale itself divides it back out, in `scaleAndCast`
(`mlx/ops_extra.go`):

```go
return Mul(out, DivScalar(scale, Nvfp4MaxProduct)).AsType(out.DType())
```

**`f32(f32(m × 2688) / 2688)` is not `m`.** Checked against the checkpoints in #312: it
returns `m` exactly for both of 12b's vision global scales and misses by one float32 ulp
for **17 of 31b's 191**. One ulp in a per-tensor scale flips bf16 roundings of that
layer's output, and 27 encoder layers grow it into something the golden test sees: on
MLX-CUDA the 31b vision encoder moved `max sampled Δ 0.0898 → 0.1094` and
`norm_mean 96.999 → 97.056` across the fold. A two-line control on `8a7ba949` that
restored the old arithmetic reproduced the pre-fold line **to every digit**, so the round
trip is the whole of the CUDA-side move.

Three wrappers apply the scale themselves, and all three take the divide:

| wrapper | why it applies the scale itself | reached by |
|---|---|---|
| `QuantizedMatmul` | `mlx_quantized_matmul` has no global-scale argument | every dense nvfp4 linear — the whole vision tower and the dense language model |
| `Dequantize` | MLX's argument accepts only a scalar; callers pass per-expert banks | weight inspection, and #287's prefill dequantisation |
| `GatherQMM` off Metal, with `rhsIndices` | the CUDA path scales gathered rows in the wrapper | nvfp4 MoE experts |

Two paths hand the scale to MLX and genuinely need the `× 2688` form: `GatherQMM` on
Metal (or without `rhsIndices`), which passes `gs` into `mlx_gather_qmm`, and the fused
`QQMM` path, whose identity scale is `Nvfp4MaxProduct` itself.

So the stored representation is chosen for the two consumers that want it, and the three
that do not want it pay a lossy round trip on every call.

## Decision

**Store `m`. Convert to MLX's representation at the call sites that hand the scale to
MLX, and nowhere else.**

- `ToMLXGlobalScale` keeps the checkpoint multiplier (cast to float32, flattened) and
  stops multiplying by `Nvfp4MaxProduct`.
- `scaleAndCast` multiplies by the scale as given and stops dividing.
- `GatherQMM`'s native branch and `QQMM` multiply by `Nvfp4MaxProduct` where they build
  the argument MLX consumes; the identity scale stays as it is.
- A test asserts the round trip: for every global scale in a served checkpoint, the value
  a wrapper applies is bit-identical to the checkpoint's `m`.

## Options considered

- **Leave it.** The error is one ulp and every bound the golden test checks still passes.
  Rejected: it is a silent, permanent divergence from the reference implementation on
  every dense nvfp4 linear, it made a pin move look like an encoder regression once
  already (#312), and it costs nothing to remove.
- **Restore the pre-fold arithmetic everywhere** (drop `× 2688` entirely). This is the
  control used in #312 and it does reproduce the old numbers, but it breaks the MoE paths
  that hand the scale to MLX. Rejected.
- **Store both forms.** Explicit, and two arrays per scale plus a rule about which to
  reach for. Rejected as more surface than the problem warrants: the conversion is one
  multiply at two call sites.
- **Keep the MLX form and make `scaleAndCast` exact** by folding `1/2688` into the
  multiply as a double. Rejected: it fixes the arithmetic at the point of use while
  leaving a stored value that is not what any checkpoint says, so the next wrapper added
  inherits the trap.

## Consequences

- On CUDA the vision encoder returns to bit-parity with the pre-fold build and with
  `mlx-vlm`, which is what the goldens in `mlxrunner/testdata` were generated against.
  The 31b golden expectations move back (`max Δ 0.1094 → 0.0898`) and should be re-taken
  in the same change.
- **#287 inherits the fix.** Its prefill path dequantises nvfp4 weights to bf16 through
  `Dequantize`, so today it would bake the ulp error into the dequantised weights for the
  whole prefill. With this decision, dequantised prefill and quantised decode apply the
  same scale, which removes one objection to that PR. It does not address the others: the
  PR still changes output deterministically and costs +6.8 GiB peak, and it stays held on
  those grounds.
- Metal is unaffected in behaviour: its `GatherQMM` still receives `m × 2688`. Its 26b and
  31b encoder numbers move for the unrelated reason in #312 (mlx#3912), and this change
  does not interact with that.
- One risk, and it is the reason for the test: a call site that hands the scale to MLX and
  is missed would silently apply a scale 2688× too small. The failure is loud in output
  quality, not in a crash.
