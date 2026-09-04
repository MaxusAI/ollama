# `format:"json"` disables speculative decoding on the MLX runner, costing 42% of decode

> **Superseded as an implementation by ADR 0033** (2026-08-28, the v0.33.2 fold):
> the fork's pure-Go constrained-sampling layer measured here was retired in favour of
> upstream's MLX grammar engine. This document is the *evidence* the ADR cites
> (`drafted=0`, the cold-start deadlock) and stays for that reason; the mechanisms it
> describes are no longer on the request path.


MaxusAI-fork reference. Measured 2026-08-17/18 on CUDA (RTX PRO 6000 Blackwell),
`maxusai/ollama:0.32.14-rc0-dynres-mlxfix` + the memory-limit override.

## The measurement

Same model family, same 1920×1080 image (`visimgs/scene_hd.png`), same prompt —
which asks for JSON in **both** arms, so the only variable is whether the
request sets `format:"json"`. Alternating arms, warmed first, `n=3`.

| engine | `format:"json"` | unconstrained | penalty |
|---|---|---|---|
| **MLX** `gemma4:31b-nvfp4` | 21.06 / 21.61 / 20.73 → **21.13** | 35.92 / 36.30 / 37.38 → **36.53** | **−42%** |
| **GGUF** `gemma4:31b-it-q4_K_M` | 57.75 / 57.12 / 56.82 → **57.23** | 57.36 / 59.14 / 58.00 → **58.17** | **−1.6%** |

`prompt_eval_count` was 1149 in all twelve runs. llama-server pays essentially
nothing for the same constraint; the MLX runner pays nearly half its decode
rate.

## The cause is structural, not the mask

`constrainedDecoder.next` (`x/mlxrunner/constrain.go:130`) returns exactly one
`sampler.Result` per call and feeds exactly one token into `forward`:

```go
out := d.pending
logits := d.forward(out.Token.ExpandDims(-1))   // one token in, one forward
...
d.pending = d.sampleMasked(logits)
return []sampler.Result{out}, nil               // one token out
```

It keeps the speculation session in lockstep — `d.spec.committed`, `d.spec.settle`
— so the draft KV stays valid, but it never drafts. **Constrained decoding is
one token per forward pass; unconstrained decoding is speculative.** The gap is
the speculation, not the constraint.

That is consistent with everything else observed:

| | speculation | tok/s |
|---|---|---|
| MLX unconstrained | active, acceptance 0.76, `avg_draft` 2–3 | 36.5 |
| MLX `format:"json"` | none | 21.1 |
| GGUF, either | llama.cpp verifies drafts against the grammar | 57–58 |

## The mask is NOT the cost, and this was measured before concluding

The first hypothesis was that `constraintBias` — which fills a `vocabDim`
float32 buffer with `-Inf`, walks the mask, and uploads a fresh device array on
**every** token — was the expense. gemma4's vocabulary is 262,144
(`embed_tokens` is `[262144, 672]`), so that is a 1 MiB memset plus a 1 MiB
host→device copy per token, and it is *not* cached even though the mask it
derives from is (`Vocab.Mask` caches on the grammar `StateKey`).

`BenchmarkConstraintBiasFill` says otherwise:

```
BenchmarkConstraintBiasFill-32    200    714719 ns/op    0 B/op    0 allocs/op
```

0.71 ms per token against a ~47.6 ms decode step at 21 tok/s: **~1.5%**. Even
with the upload and the 262k-element add it cannot approach 42%. The hypothesis
was wrong and the benchmark is what disproved it — which is the reason both
benchmarks are committed alongside this note rather than deleted.

## What would fix it

Speculative decoding *under* a grammar: draft k tokens, check each against the
matcher, accept the longest legal prefix, roll back the rest. That is what
llama.cpp does and why its penalty is 1.6%. It is a feature, not a tweak, and it
is not attempted here.

**Update 2026-08-18:** it has since been attempted, gated behind
`OLLAMA_MLX_GRAMMAR_SPECULATION`, and measured. It produces correct constrained
output but no speedup, because the depth controller cold-starts at 0 and cannot
leave it, so it never drafts. The penalty above is unchanged and the gate ships
off. See [grammar-aware speculation is correct and inert](grammar-speculation-measured-inert.md).

A smaller, independent improvement remains available: cache the bias array
under the same `StateKey` the mask cache already uses. Worth ~1.5% on its own,
but it also removes 1 MiB of device allocation **per token** from a path that
runs for every constrained request — and MLX's buffer cache was separately
measured holding 13.16 GiB on this backend, which that churn feeds.

## What this does and does not change

- **Does not** invalidate any published number. Every vision-suite figure for
  both engines was measured under `format:"json"`, so the MLX-vs-GGUF comparison
  remains apples-to-apples.
- **Does** change how those figures should be read. "MLX decodes at 27 tok/s" is
  a statement about *constrained* decode. Unconstrained, the same model on the
  same host runs 36.5.
- **Does** re-attribute the engine gap. Constrained, MLX-vs-GGUF is 2.7×.
  Unconstrained it is 1.6×. So roughly 60% of the deficit is this one behaviour,
  and the rest is the genuine engine difference.

## Provenance and limits

- `n=3` per arm, one model per engine, one image, one prompt. Enough to act on;
  not a campaign.
- The MLX arm's `prefill` and `avg_draft` columns are unusable: the repeated
  identical prompt was served from the prefix cache, so `prompt_eval_duration`
  ≈ 0 (reported prefill 5×10⁷ tok/s) and there was little left to draft. The
  GGUF arm showed sane prefill (~1000 tok/s), so the two engines were not in
  identical cache states. The **decode** comparison is computed within each
  engine, which is what the penalty figures rest on; a cross-engine absolute
  wants fresh prefixes per request.
- Not run through `run_engine_compare.sh`: this is an engine-behaviour probe,
  not a scored arm, and it varies a request field the suite holds fixed. If any
  of it is promoted to a published table it belongs behind the runner
  (SPEC H1/H3).
