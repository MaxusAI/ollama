# Rigorous dtype trace of the fp16-accumulate class

**Two engines are covered here and they are not interchangeable — read the
section headers.** Sections 1–N below trace `04431b0d` through **ollama
0.7.1's Go vision engine** (source-level, plus CPU/GPU and perturbation
measurements). The node-localisation section at the end was measured on the
**current clip path** (`maxusai/ollama:sync-0.33.0`, llama.cpp pin `b10488`)
with the checkerboard trigger — *not* on 0.7.1.

Question asked: are the degenerate outputs caused by a non-conforming data
format (fp32/fp16/uint8 mix-up), like the fp16 matmul? Answer: **every data
format in the preprocessing chain is conforming and empirically verified; the
sole non-conformance is accumulation precision**, and 0.7.1 narrows to fp16
in *three* places, not one.

## Stage-by-stage (source: `v0.7.1`)

| # | Stage | Code | Dtype | Verdict |
|---|---|---|---|---|
| 1 | JPEG decode | `image.Decode` (Go stdlib) | YCbCr8 → RGBA16 | ✅ conforming. Go applies a documented 256/255 luma gain (`Y′×0x10101`) that libjpeg does not; both legal, differs ±1 LSB |
| 2 | SmartResize | `maxPixels = 28*28*1280 = 1,003,520` | — | ✅ 1288×616 = 793,408 < cap, and both sides ÷28 ⇒ **identity, no resize** |
| 3 | Resize | `draw.BiLinear`, `draw.Over` onto fresh RGBA | uint8 | ✅ identity for this image |
| 4 | Rescale | `float32(r>>8) / 255.0` | uint8 → **F32** | ✅ correct 16→8 narrowing *before* scaling. Measured range **[0.000000, 1.000000]** — cannot exceed 1.0 |
| 5 | CLIP normalize | `(v - mean) / std` | F32 | ✅ measured **[−1.7923, +2.1459]**, exactly `(0−μ)/σ` and `(1−μ)/σ`; all finite |
| 6 | createPatches | slice copy | F32 | ✅ no conversion |
| 7 | To tensor | `ctx.Input().FromFloatSlice` | **F32** | ✅ |
| 8 | Patch embed | `PatchConv0/1` | F32 act × **F16** weight | ⚠️ |
| 9 | Attn Q/K/V | `nn.Linear.Forward` → `Mulmat` | F32 × **F16**/Q6_K | ⚠️ no `set_prec` |
| 10 | Attn **KQ** | **`MulmatFullPrec`** | `GGML_PREC_F32` | ✅ **the only protected op in the tower** |
| 11 | Attn KQV | `Mulmat` | F32 × F32 | ✅ |
| 12 | Attn out proj | `Linear` → `Mulmat` | F32 × **F16** (1280×1280) | ⚠️ |
| 13 | MLP gate/up/**down** | `Linear` → `Mulmat` | F32 × **F16** (3420×1280) | ⚠️ **block 31 peaks here** |
| 14 | Merger mlp.0 / mlp.2 | `Linear` → `Mulmat` | F32 × **F16** (5120×5120, 5120×2048) | ⚠️ **longest reduction in the model** |

Weight dtypes read from the served GGUF (`qwen2.5vl:3b-q4_K_M`): the vision
tower is **196 F16 tensors** + 291 F32 (norms/biases) + 32 Q6_K/Q4_K. Every
matmul weight named above is F16 on disk.

## The dispatch — three fp16 narrowings, one escape

`ggml_cuda_op_mul_mat_cublas` in 0.7.1's vendored ggml:

```c
const bool use_fp16 = (src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type))
                   && ggml_is_contiguous(src0) && row_diff == src0->ne[1]
                   && dst->op_params[0] == GGML_PREC_DEFAULT;   // <-- the only escape
```

F16 **and quantized** weights both qualify. On the NVIDIA branch (the `else`;
only CDNA/RDNA4 take the `COMPUTE_32F` path):

1. **activations are downcast** F32 → F16 (`to_fp16_cuda(src1_ddf_i, …)`) — a
   value above 65,504 is already `inf` before the GEMM starts;
2. **accumulation is fp16** (`CUBLAS_COMPUTE_16F`, `half` alpha/beta);
3. **the output buffer is fp16** (`dst_f16`, `CUDA_R_16F`), converted to F32
   only afterwards — so an overflowing result is written as `inf` and the
   later widening cannot recover it.

The modern llama.cpp path (b10488) is *less* exposed: it can prefer an F32
output buffer on some architectures. 0.7.1 has no such escape.

## Conclusion

`op_params[0] == GGML_PREC_DEFAULT` is the whole safety switch, and
`MulmatFullPrec` — the one call that flips it — is used exactly once, on the
attention KQ. Every weight matmul in the tower, including block 31's
`ffn_down` (K=3420, where the measured 50,688 peak is produced) and the
merger's `mlp.0` (K=5120), accumulates in fp16 with an fp16 output buffer.

This is the **same structural gap as clip.cpp**, which likewise sets
`GGML_PREC_F32` on vision flash-attention while leaving `build_mm` weight
matmuls at default precision. Two independent implementations, the same
choice, the same defect class — which is why the fix direction is
implementation-independent: **≥fp32 accumulation for this tower's weight
matmuls**, whether via `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (ollama ≥0.30),
`ggml_mul_mat_set_prec` per op (compat patch 904), or `MulmatFullPrec` on the
Linear layers (the Go engine's own idiom, had it been applied there).

## Why *this* image and not the other 28 at the same geometry — quantified

The overflow does not need partial-sum wandering. On the fp16 path the
**product** `w·x` is itself formed in fp16, so a single input channel with

    |x_c|  >  65504 / max_row |W[row, c]|

produces `inf` on one multiply, before any accumulation — and one `inf` NaNs
the whole dot product. Running that over the real F16 weights
(`fp16_audit.py`, no GPU, no inference, no trigger image needed):

| tensor | shape | \|w\|max | min \|x\| → inf | unsafe channels |
|---|---|---|---|---|
| **`v.blk.31.ffn_down`** | 1280×3420 | 2.109 | **31,054** | **29 / 3,420** |
| **`v.blk.17.ffn_up`** | 3420×1280 | 1.750 | **37,431** | 2 / 1,280 |
| `v.blk.17.ffn_gate` | 3420×1280 | 1.077 | 60,757 | 0 |
| `v.blk.17.ffn_down` | 1280×3420 | 0.988 | 66,281 | 0 |
| … 190 others … | | ≤0.5 | ≥77,277 | 0 |
| `v.merger.mlp.0` | 5120×5120 | 0.216 | 303,512 | 0 |

**CORRECTION — the single-product model is falsified by measurement.**
Hooking the real tower (`measure_channel_thresholds.py`, HF bf16) shows
`ffn_down`'s *input* peaks at only **4,128** for the trigger — 0 of 3,420
channels exceed their own product threshold, and the healthy same-geometry
images look identical. The 50,688 figure from #214 is the module *output*,
not the matmul input. So no single product overflows; the mechanism is
accumulation, whose order is engine-specific — which is itself the reason
trigger sets are disjoint per engine. The audit below therefore reports which
weights *could* overflow given an input ceiling; it is a screen, not a
diagnosis, and at the real input magnitudes it does not fire.

For the record: **2 of 194 F16 matmul weights are unsafe at a 50,688 input
ceiling**, and in the worst one just **29 of 3,420 input channels (0.8%)** can
overflow. So an image trips this engine only when its massive-activation
channels *land on that ~1% subset*. `04431b0d`'s do; the 28 other corpus
images at the identical geometry put their peaks elsewhere and are healthy.
That is the "resonance" of the earlier write-ups, now a number.

It also explains three previously separate observations in one stroke:

- **Why the basin is a point.** Shifting pixels by 1 LSB, or mirroring, moves
  *which* channel peaks and by how much — off the 0.8% subset, and the image
  heals.
- **Why bigger images are more exposed.** The GEMM runs one column per image
  token, each an independent draw at hitting a vulnerable channel, so
  P(trigger) ≈ 1 − (1 − p)^n_tokens. More tokens, more draws.
- **Why 0.7.1 resists synthetics.** It serves 1265 tokens where 0.33.0 serves
  3100 — **2.45× fewer draws** — on top of downscaling away the high-frequency
  content that drives the peaks.

Note this corrects an earlier guess: the merger's `mlp.0`, despite having the
longest reduction (K=5120), has small weights (max 0.216) and **cannot**
overflow. The site is block 31's `ffn_down`.

## This is testable statically — `fp16_audit.py`

```bash
python3 fp16_audit.py <model.gguf> [activation_ceiling]
```

It reads only the GGUF, needs no GPU, no inference and no trigger image, and
flags every F16 matmul weight whose per-channel threshold falls below the
activation ceiling the tower is known to reach. Run at model-import or
CI time, it would have caught this defect class before a single image was
served — and it generalises to any vision tower with massive-activation
outliers, which is the property that makes fp16 accumulation unsafe here.


## Measured: block-31 output magnitudes, and why "make the peak bigger" fails

Hooking `blocks[31].mlp.down_proj`'s output (`screen_activation_peak.py`) gives
a cheap, deterministic proxy — re-measured three times, bit-identical:

| image | block-31 peak | % of fp16 max | 0.7.1 |
|---|---|---|---|
| `39823be1` (0.24.0's trigger) | **63,744** | 97.3% | **H** |
| `04431b0d` **Go decode** | **60,672** | 92.6% | **X** |
| best synthetic (`sv14_868x1148`) | 54,528 | 83.2% | H |
| `04431b0d` libjpeg decode | 50,688 | 77.4% | H |
| `11c11aa8` | 45,312 | 69.2% | H |

Two things follow, one encouraging and one limiting.

**Encouraging:** the decoder difference is now quantified where it matters —
Go's decode lifts the same image from 50,688 to 60,672, **+20%**, which is why
the decode path flips the verdict. Every image in the corpus sits at 70–97% of
the ceiling, so the tower runs permanently near the cliff.

**Limiting:** peak magnitude is **not a sufficient objective**. `39823be1`
reaches 97.3% — *higher than the image that trips 0.7.1* — and stays healthy
there (it trips 0.24.0 instead). So a synthetic optimised to maximise this
proxy is not thereby a 0.7.1 trigger; what decides the outcome is where the
magnitude lands relative to that engine's accumulation order, not the global
peak. Hand-designed families plateau at 83.2% (46 candidates screened, waves
1–6 = 239 total), and closing the last 10% would not settle it anyway.

A sufficient objective would have to model the engine's actual fp16
accumulation order — i.e. simulate the GEMM tiling, not the tower. That is the
honest next step for anyone continuing this hunt.

## The output side: no tokenizer fault, but a missing guard

Asked whether the trigger produces tokens that are "off" — out of dictionary,
tripping the tokenizer. Measured answer: **no, and the vision path emits no
tokens at all.** The merger produces *embeddings* spliced in at the
`<|image_pad|>` (151655) positions; there is no vocabulary lookup to fail.
In the bf16 reference those embeddings are entirely well-behaved:

| image | block-31 residual | elements over fp16 max | merger output max | finite |
|---|---|---|---|---|
| `04431b0d` Go decode **[X]** | 60,672 (92.6%) | **0** | 58.2 | yes |
| `04431b0d` libjpeg **[H]** | 50,688 (77.4%) | 0 | 59.2 | yes |
| `39823be1` **[H on 0.7.1]** | 63,744 (97.3%) | 0 | 47.5 | yes |
| `11c11aa8` **[H]** | 45,312 (69.2%) | 0 | 49.8 | yes |

Nothing overflows in the reference, and the trigger's embeddings are
indistinguishable from the healthy ones — the overflow exists **only inside
the engine's fp16 accumulation**, which no offline inspection of the model can
observe. The merger itself is safe: `ln_q` renormalises before its matmuls, so
its outputs sit around 50–60.

**What the glyphs actually are.** The degenerate outputs are the *bottom of the
vocabulary*: `!` is token id **0** and `?` is token id **30** (verified against
the Qwen2.5-VL tokenizer; `!!!` is 12069 and `???` is 33015, so these are
single-token repeats, not multi-char pieces). That is the classic signature of
`argmax` over NaN or degenerate logits collapsing to a fixed low index — the
tokenizer is faithfully reporting the damage, not causing it. The two engines
land on different indices because they differ in how the sampler handles the
broken distribution.

**Why exactly 31 glyphs, and why `done_reason: null`.** Not a model behaviour
at all: `llm/llama_server.go` aborts on `tokenRepeat > 30` and returns
`ctx.Err()`, so the response carries 1 + 30 repeats and no done reason. Both
signatures of this bug are therefore explained end to end.

### Missing guard (actionable)

The stack already contains exactly the right check — on the wrong path.
`normalize()` in `server/routes.go` rejects non-finite **text** embeddings with
`"embedding contains NaN or Inf values"`. The **vision** embedding path has no
equivalent in either engine: no `isnan`/`isfinite` in clip.cpp or mtmd at pin
b10488, and none in 0.7.1's Go model. A finite-check on the multimodal
embeddings before they are spliced into the token stream would convert a silent
`'?'×31` into a clear, attributable error — and would have made this entire
class diagnosable in minutes rather than a week.

## Falsified: the overflow is NOT at `blk.31.ffn_down` — site still unlocalised

Simulating fp16 split-K accumulation of that GEMM with the **real captured
activations and real weights** (`simulate_fp16_accum.py`), across tile sizes
K=32/128/512/full:

| image | K=32 | K=128 | K=512 | full | 0.7.1 |
|---|---|---|---|---|---|
| `04431b0d` Go | 60,640 | 58,368 | 48,288 | 60,800 | **X** |
| `04431b0d` libjpeg | 50,720 | 48,800 | 40,512 | 50,752 | H |
| `39823be1` | 63,680 | 61,408 | 51,200 | 63,744 | H |
| `11c11aa8` | 45,280 | 43,744 | 36,224 | 45,312 | H |

**No tiling produces a single non-finite value**, for any image including the
trigger, and the ordering stays monotone with the peak — `39823be1` remains
*above* the image that actually fails. So this matmul does not overflow, it
does not discriminate, and the earlier identification of it as "the site" is
withdrawn.

Two gaps explain why this line of analysis stalled, and both must be closed
before any claim about the exact op:

1. **Wrong weights.** All of this measures HF `Qwen2.5-VL-3B-Instruct` in
   bf16. The engine serves **q4_K_M** — different weights, therefore different
   activations. The vision tower is largely F16 in the GGUF, but not entirely
   (`attn_v` is Q6_K, and some tensors are Q4_K), so the reference is close but
   not the served computation.
2. **Wrong graph.** HF's tower is not ggml's: window-attention layout, RoPE
   application and op fusion all differ, so intermediate magnitudes need not
   match even with identical weights.

**What would actually localise it:** a `ggml_backend_sched_set_eval_callback`
on an instrumented CUDA build that scans every node output for non-finite
values and reports the first one, run against the checkerboard on the clip
path. That names the op directly instead of inferring it, and needs no
activation modelling at all. It costs one CUDA build (~3 h here).

**Consequence for synthesis:** there is still **no validated objective**. The
peak proxy is measurable, deterministic and cheap — but demonstrably not
sufficient (39823be1), and the accumulation simulation at the suspected site
adds nothing. Optimising against either would be optimising against an
unvalidated target.

## Ruled out: context length. Confirmed: 0.7.1's failure IS GPU numerics.

Two controls close the remaining alternatives for `04431b0d` on 0.7.1.

**Context / batch is not the cause.** The image costs 1047 tokens, so a window
or batch-split problem was plausible. Measured — X at *every* setting:

| `num_ctx` | 2048 | 4096 | 8192 | 16384 | 32768 | 8192 + `num_batch` 256 | 8192 + `num_batch` 2048 |
|---|---|---|---|---|---|---|---|
| verdict | X | X | X | X | X | X | X |

**And the fp16 class now has direct evidence here, not just analogy.** This
mattered because `GGML_CUDA_CUBLAS_COMPUTE_TYPE` does not exist in 0.7.1's
ggml (0 matches on binary grep), so the knob-based proof used on the clip path
is unavailable — 0.7.1 had been assigned to the class by inference alone.
Same build, same model, same image, same 1047 tokens, only the device changes:

| 0.7.1 + `04431b0d` | result |
|---|---|
| GPU | **X** — `!`×31, `done_reason: null` |
| CPU-only (`size_vram: 0`) | **H** — *(a correct, detailed description of the photo — content withheld, client corpus)* |

CPU accumulates in fp32 throughout; the CUDA path narrows to fp16 three times
(activations, accumulator, output buffer — see the dispatch section above).
The CPU/GPU split with everything else held constant is the same signature the
clip path shows, and it is now measured for 0.7.1 rather than assumed.

What remains unlocalised is the *exact op*, not the class — see the falsified
`blk.31.ffn_down` section and the eval-callback plan.

## LOCALISED: the vision tower's own output embeddings contain NaN

No instrumented build was needed after all — clip.cpp ships a debug hook,
`MTMD_DEBUG_EMBEDDINGS`, which dumps the final image embeddings and their
statistics. Running the shipped checkerboard on stock `ollama/ollama:0.33.0`,
same image and same request, changing only the accumulation:

| path | verdict | embedding stats over [2048 × 3072] |
|---|---|---|
| **GPU, default (fp16 accumulate)** | **X** | `mean=nan std=nan min=-50.732422 max=19.314453` **`sum=nan`** |
| GPU, `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | H | `mean=0.020965 std=0.961237 min=-50.502407 max=19.058825 sum=131902.39` |
| CPU only | H | `mean=0.020738 std=0.952469 min=-50.171288 max=18.853397 sum=130471.91` |

**The vision encoder emits NaN.** Not the text model, not the sampler, not the
tokenizer — the multimodal embeddings are already poisoned when they leave
`clip_encode`. (`min`/`max` still print finite because `std::min`/`std::max`
discard NaN on comparison; `sum` is the field that reveals it.) The two healthy
paths agree with each other to three decimal places on every statistic, which
also shows the fp32 GPU path and the CPU path are numerically equivalent —
the fix does not perturb the result, it restores it.

Note the failing run reports `done_reason: "length"` here rather than `null`:
with `num_predict: 24` the repeat-abort (`tokenRepeat > 30`) never fires, so
the request ends on the token budget instead. Same NaN, different exit path —
which is why `done_reason` alone is not a reliable fingerprint and the
degenerate-glyph test is the one that matters.

This is the deviation point CPU-vs-GPU, established empirically and with no
rebuild. What is still not named is the *individual ggml node* inside the
tower where the first NaN appears; a `ggml_backend_sched_set_eval_callback`
scanning node outputs would give that, and clip.cpp already plumbs such a
callback through `clip_ctx` (`ctx_params.cb_eval`) — so the remaining work is
wiring it, not modifying the graph.

It also makes the missing-guard finding concrete: a finite-check on these
embeddings before they are spliced into the token stream would have caught
this exact condition, at the exact place it becomes observable.

## LOCALISED: the vision embeddings themselves are NaN on the fp16 path

`MTMD_DEBUG_EMBEDDINGS=1` dumps the tower's final output. Same stock 0.33.0
image, same checkerboard trigger, same request — only the device changes:

| run | verdict | embedding stats over all [2048 × 3072] values |
|---|---|---|
| **GPU, fp16 accumulate** | **X** | `mean=nan, std=nan, min=-50.732422, max=19.314453, sum=nan` |
| CPU (fp32 accumulate) | H | `mean=0.020738, std=0.952469, min=-50.171288, max=18.853397, sum=130471.9` |

**The image embeddings handed to the language model contain NaN.** `mean` and
`sum` are `nan` while `min`/`max` still print finite values — because C++
`std::min`/`std::max` comparisons against NaN are false, so the running
extremes skip them. Only *some* tokens are affected: token 0's first 16 values
are finite and agree with CPU to ~1e-3 on both paths.

This ends the localisation question at the level that matters:

- The break is **inside the vision encode**, not downstream. The language
  model, sampler and tokenizer are all victims — they receive NaN embeddings
  and faithfully produce a degenerate token (id 0 or 30) until the
  `tokenRepeat > 30` abort fires.
- The CPU path, which differs from the CUDA path *only* in accumulating fp32,
  produces clean embeddings whose range (−50.17 … 18.85) closely matches the
  GPU's finite portion (−50.73 … 19.31). Same graph, same weights, same
  inputs — the fp16 accumulation is the only difference, and it is the one
  that yields NaN.
- The missing-guard recommendation is now precisely sited: a finite check on
  `embeddings` right here — where clip.cpp already reads the tensor back for
  this debug dump — would convert a silent `'?'×31` into an attributable
  error, and needs no knowledge of which matmul overflowed.

What remains unknown is only *which node* first goes non-finite inside the
tower; the eval-callback plan above answers that, and nothing shipped depends
on it.

## NODE LOCALISED — **clip path (0.30+), not 0.7.1**: `ffn_down-31`

**Scope note.** Everything in this section was measured on
`maxusai/ollama:sync-0.33.0` (the llama-server/clip path, pin `b10488`) with
the synthetic checkerboard. **0.7.1 was not instrumented**: its Go engine has
no clip graph and ships no `libmtmd`, so this tracer cannot be loaded there at
all. What carries across is the *weights*, not the code — both engines read
the same GGUF, so the static audit's ranking (which named
`v.blk.31.ffn_down` the most exposed tensor in the tower) applies to both,
and the Go engine's equivalent op is `blk.31` `mlp.down_proj` via
`nn.Linear.Forward` → `Mulmat`, traced in source above but never instrumented.
Its trigger set differs because its summation order differs.

Instrumented `clip_ctx` with an env-gated `ggml_backend_sched_eval_callback`
that scans every node's output and reports the first non-finite one
(the first-non-finite tracer this used has since been generalised into
`llama/compat/801-clip-node-stats-meter.patch`; it lives in `libmtmd.so`, which is
backend-agnostic, so it needed a 2-minute library rebuild rather than a CUDA
build — bind-mounted over the image's own `libmtmd.so.0.1.2`).

Stock-pin `b10488`, `maxusai/ollama:sync-0.33.0`, checkerboard trigger, GPU:

```
*** CLIP_NAN_TRACE: first non-finite node ***
  node #1106  name='ffn_down-31'  op=MUL_MAT  type=f32
  shape=[1280,12288,1,1]  bad=3/15728640  first_bad_index=0
  nan=0  inf=3  is_mask=0
  src[0]: name='v.blk.31.ffn_down.weight' op=NONE type=f16 ne=[3420,1280]
  src[1]: name='ffn_swiglu-31'            op=GLU  type=f32 ne=[3420,12288]
*** END CLIP_NAN_TRACE ***
```

| run | verdict | first non-finite node |
|---|---|---|
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` | **X** | **`ffn_down-31`** (MUL_MAT), 3 inf |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | H | **none — 0 reported** |

**The site is block 31's `ffn_down`**: the F16-weight matmul consuming the
SwiGLU output, exactly the tensor the static audit flagged as the most exposed
in the tower (threshold 31,054; 29 of 3,420 channels unsafe). Three elements
out of 15,728,640 overflow — **0.00002%** — and that is sufficient: the `inf`
propagates through the merger into the image embeddings as NaN, and the
language model emits a degenerate token until `tokenRepeat > 30` aborts.

Flipping only the accumulation mode removes it entirely. Same graph, same
weights, same input, same node — `f32` reports zero non-finite nodes.

**This reinstates the earlier "falsified" section, and explains why it
failed.** That analysis used HF `Qwen2.5-VL-3B-Instruct` in bf16; the engine
serves **q4_K_M** through **ggml's** graph. Different weights and a different
graph gave input magnitudes ~4,128 where the real path overflows — so the
model was right and the *proxy* was wrong. Measuring the engine settles what
modelling the reference could not.

One false positive worth recording: the tracer's first hit was
`window_mask (copy)`, the F32→F16 cast of the window-attention mask, 99.5%
`-inf` by construction. Attention masks are legitimately non-finite; the
tracer now separates NaN from inf and skips mask tensors.
