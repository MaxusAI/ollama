# MLX vision: ecosystem survey, and sizing parity with mlx-vlm

MaxusAI-fork reference (fork-only; does not exist upstream). Written 2026-08-07.

Two questions, answered in one pass:

1. **Has anyone else built MLX vision for the three arches this fork serves** — gemma4,
   qwen, nemotron?
2. **Does our Go port agree with mlx-vlm on image sizing?** A divergence there is
   exactly what would make the MLX engine and llama-server disagree on identical
   inputs.

**Answers: yes to all three arches (nemotron worst-served), and our sizing is
byte-identical to mlx-vlm on every non-degenerate input tested — with two real
differences at the edges, one of them unverified.**

**Third answer, not originally asked but the most consequential:** upstream ollama has
**three open PRs** covering the same ground as this fork's gemma4 work, including a
maintainer's gemma4 MLX vision PR open since 2026-07-31. Two of them conflict with each
other on `x/mlxrunner/`. See §2. Sizing remains the one part nobody else has touched.

## 1. What this fork actually has

[`x/models/gemma4/vision.go`](../../x/models/gemma4/vision.go) is a **Go port of
mlx-vlm's gemma4 vision**, not an independent design. Its own header:

> Vision support for Gemma 4 MLX checkpoints, ported from mlx_vlm (main branch)
> `models/gemma4/vision.py` and `models/gemma4_unified`.

It covers both lineages the model card describes:

| lineage | sizes | architecture |
|---|---|---|
| `gemma4_unified_vision` | 12B | **encoder-free** — LayerNorm → Linear over flat 48px patches → LayerNorm → learned per-axis position embeddings → LayerNorm |
| `gemma4_vision` | 26B / 31B | 27-layer encoder over 16px patches, 2D RoPE attention (scale 1.0), 3×3 average pooling + standardization |

Both project into the text hidden size through `embed_vision` (weightless RMSNorm →
linear). Tests: `vision_test.go`, plus `x/mlxrunner/vision_e2e_test.go` and
`vision_golden_test.go`.

**The one deliberate deviation**, stated in the header:

> Image sizing follows ADR 0008's budget-fill ladder via `llm.BudgetFillSize` —
> grids are always exact 48-multiples, so the padding/masking paths of the reference
> are deliberately not ported.

That deviation is what §3 tests.

## 2. Ecosystem survey

### gemma4 — well served

| project | what it is |
|---|---|
| [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) | **Day-0** Gemma 4 support in v0.4.3 — vision, audio, MoE. Dedicated [`models/gemma4/`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md) module, SigLIP2 encoder shared across variants. **This is our upstream reference.** |
| [VincentGourbin/gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx) | Native text+vision+audio+**video** via MLX Swift |
| [FakeRocket543/mlx-gemma4](https://github.com/FakeRocket543/mlx-gemma4) | PLE-safe quantization (E2B/E4B/26B/31B); patches `models/gemma4/language.py` |
| [korale77/mlx-vlm-falcon](https://github.com/korale77/mlx-vlm-falcon) | Falcon Perception + Gemma 4 for grounded reasoning |
| [mlx-optiq](https://mlx-optiq.com/docs/gemma-4) | Mixed-precision quants with image input |

### qwen — well served

mlx-vlm supports **Qwen3-VL**. [lmstudio-community](https://huggingface.co/lmstudio-community/Qwen3-VL-32B-Thinking-MLX-4bit)
ships 4-bit MLX builds of Qwen3-VL-32B-Thinking and 30B-A3B-Instruct.
[waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx) serves Qwen-VL and LLaVA
with continuous batching. Qwen 3.5 vision weights load in both mlx-lm (text-only) and
mlx-vlm (vision+text).

### nemotron — the weak spot, and it is our arch

mlx-vlm's "native support" was **silently broken**:
[issue #1090](https://github.com/Blaizzy/mlx-vlm/issues/1090) — after a model commit
stripped `auto_map`, `processor.image_processor` returns `None`, so `prepare_inputs()`
falls through to a **text-only path and ignores images entirely**. No error raised.
Reported 2026-04-29 by Robaloman; closed via PR #1177, with the interim workaround
being to pin an older model commit.

Same failure *class* the fork hit repeatedly on the llama.cpp side: a vision path that
looks wired up and quietly is not. Worth remembering as a standing risk for any
nemotron MLX work here.

### upstream ollama itself — three PRs in flight, all overlapping this fork

Checked 2026-08-07 against `ollama/ollama`. **Every one of the fork's gemma4
contributions has an upstream counterpart currently open.**

| PR | author | opened | size | subject | fork counterpart |
|---|---|---|---|---|---|
| [#17487](https://github.com/ollama/ollama/pull/17487) | **@dhiltgen** (maintainer) | 2026-07-31 | +2394/−25 | **mlx: add Gemma4 vision support** | `x/models/gemma4/vision.go` |
| [#17600](https://github.com/ollama/ollama/pull/17600) | **@jessegross** (maintainer) | 2026-08-07 | +2014/−115 | **mlxrunner: Add image input support** | the MLX media plumbing |
| [#17154](https://github.com/ollama/ollama/pull/17154) | @yatishgoel | 2026-07-13 | +323/−0 | **llm: raise and expose Gemma 4 image token budget** | ADR 0003's `image_{min,max}_tokens` |

**The two maintainer PRs conflict with each other.** #17600 implements **Qwen 3.5/3.6
only** (`x/models/qwen3_5/vision.go`); it touches `x/create/gemma4.go` but not gemma4
vision. #17487 does gemma4 (`x/models/gemma4/gemma4.go`). They collide on
**`x/mlxrunner/client.go`, `pipeline.go`, `runner.go`**, and they take *different*
architectural approaches to the same layer:

- jessegross: `x/mlxrunner/media.go` + `x/mlxrunner/model/base/media.go`
- dhiltgen: `x/mlxrunner/multimodal.go` + `x/mlxrunner/model/base/media_cache.go`

Whichever lands first sets the interface any downstream port must match. As of
2026-08-07 both are open and neither has review comments.

**The user-facing bug is open and widely confirmed.**
[Issue #17065](https://github.com/ollama/ollama/issues/17065) (opened 2026-07-07 by
@ytooyama, 15 comments) — MLX vision models "do not appear to receive image input",
reproduced across `gemma4:12b-mlx`, `gemma4:31b-mlx` and `qwen3.5:4b-mlx`. The model
sees only the `[img-0]` placeholder and reports it cannot see the image; the GGUF build
of the same model works. jessegross's own PR description states the cause plainly:

> MLX vision checkpoints are already exposed as image-capable, but the client does not
> send media to the runner and the prompt is processed as ordinary text.

**This fork already fixed that** — it is what `x/models/gemma4/vision.go` is for.

### On sizing specifically, still nobody

No MLX project — **including the three upstream PRs** — addresses the budget ladder
snap, budget-fill scaling, or the vertical coordinate error that
`llama/compat/004` fixes. #17154 raises and exposes the budget (the same ground as
ADR 0003) but does not touch sizing geometry. mlx-vlm's gemma4 README documents usage,
not preprocessing; the nearest adjacent fix in its release notes is "gemma4 multi-image
processing for different-sized images".

> **Scope correction.** An earlier draft of this document concluded "nobody else
> addresses the sizing problem" and implied the fork was alone on gemma4 MLX vision
> generally. The first half stands; **the second does not** — @dhiltgen has had an open
> gemma4 MLX vision PR since 2026-07-31. What remains distinctive is the *sizing*, not
> the vision port.

**Not in scope, despite the name:** [mlx-optiq](https://pypi.org/project/mlx-optiq/)
(v0.4.18, 2026-08-07, MIT) is a **mixed-precision quantization optimizer** — per-layer
sensitivity quantization, SSD expert streaming, mixed-precision KV, speculative
decoding, distributed inference. It supports Gemma 4 *with image input*, so it overlaps
on **which model** but not **which layer**. Quantization operates on weights; our defect
is in preprocessing geometry, upstream of any weight. Applying it would not change bbox
behaviour.

## 3. Sizing parity check — the decisive comparison

### The two implementations

**mlx-vlm**, `Gemma4ImageProcessor.aspect_ratio_preserving_resize` in
`mlx_vlm/models/gemma4/processing_gemma4.py`:

```python
target_px = max_patches * (patch_size**2)      # max_patches = max_soft_tokens * pooling_kernel_size**2
factor    = math.sqrt(target_px / (height * width))
side_mult = pooling_kernel_size * patch_size   # 3 * 16 = 48

target_height = int(math.floor(factor * height / side_mult)) * side_mult
target_width  = int(math.floor(factor * width  / side_mult)) * side_mult
```

**Ours**, `llm.BudgetFillSize` ([`llm/llama_server.go`](../../llm/llama_server.go)):

```go
budget     := gemma4SnapBudget(maxTokens)
pxPerToken := align * align                    // 48 * 48
factor     := math.Sqrt(float64(budget*pxPerToken) / (float64(width) * float64(height)))
wBar := max(align, int(math.Floor(float64(width)*factor/float64(align)))*align)
hBar := max(align, int(math.Floor(float64(height)*factor/float64(align)))*align)
for (wBar/align)*(hBar/align) > budget {
    if wBar >= hBar { wBar -= align } else { hBar -= align }
}
```

### The formulas are algebraically identical

`max_patches × patch_size²` = `(tokens × pooling²) × patch²` = `tokens × (3×16)²` =
`tokens × 48²` = our `budget × align²`. Same ladder `(70, 140, 280, 560, 1120)`, same
`sqrt` fill, same floor-to-48.

**Both upscale.** Neither clamps `factor ≤ 1`; neither returns early for under-budget
images. Small images are scaled *up* to fill the budget in both.

### Measured: zero divergence on normal inputs

Replicating both exactly, at budget 1120:

| input | mlx-vlm | ours | agree |
|---|---|---|---|
| 1920×1080 | 2112×1200 | 2112×1200 | ✓ |
| 1568×1568 | 1584×1584 | 1584×1584 | ✓ |
| 1280×960 | 1824×1344 | 1824×1344 | ✓ |
| 256×144 | 2112×1200 | 2112×1200 | ✓ |
| 3072×1728 | 2112×1200 | 2112×1200 | ✓ |
| 3200×32 (100:1) | 16032×144 | 16032×144 | ✓ |
| 8000×200 | 10128×240 | 10128×240 | ✓ |
| 4096×64 | 12816×192 | 12816×192 | ✓ |
| 64×4096 (portrait) | 192×12816 | 192×12816 | ✓ |
| 96×12000 (125:1) | 96×17952 | 96×17952 | ✓ |

**0/12 differ**, including panoramas up to 125:1. The header's claim — that skipping the
padding/masking port is safe because grids are always exact 48-multiples — holds for
every non-degenerate input tested.

## 4. Two real differences, both at the edges

### 4a. Extreme aspect: mlx-vlm degenerates to a zero axis

Our `max(align, …)` clamp and shave-loop have no counterpart in mlx-vlm's main path.
The clamp fires when **`w/h > budget`** — reachable at low rungs, not just exotic
aspects:

| budget | input | aspect | mlx-vlm main path | ours | ours grid |
|---|---|---|---|---|---|
| 70 | 3200×32 | 100:1 | **3984×0** | 3360×48 | 70 |
| 70 | 800×8 | 100:1 | **3984×0** | 3360×48 | 70 |
| 140 | 3200×16 | 200:1 | **8016×0** | 6720×48 | 140 |
| 140 | 4800×32 | 150:1 | **6912×0** | 6720×48 | 140 |
| 1120 | 12000×8 | 1500:1 | **62208×0** | 53760×48 | 1120 |

Ours lands exactly on budget. mlx-vlm's main path produces a zero axis and hands off to
a degenerate fallback:

```python
target_width  = min(int(math.floor(width / height)) * side_mult, max_side_length)
target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)
```

**This fallback was not evaluated** — `max_side_length` was not read. It is a different
mechanism from our clamp-and-shave, so the two *may* disagree here. **This is the one
place a Go/Python divergence is plausible, and it is unverified rather than proven
safe.**

Note a 100:1 panorama at budget 70 is an ordinary request, not a contrived one.

### 4b. Off-ladder budgets: reject vs snap

mlx-vlm **raises**:

```python
if max_soft_tokens not in _SUPPORTED_SOFT_TOKENS:
    raise ValueError(f"`max_soft_tokens` must be one of {_SUPPORTED_SOFT_TOKENS}, got {max_soft_tokens}.")
```

Ours **snaps down** via `gemma4SnapBudget`, matching llama.cpp's 004 patch. So the
fork's two engines agree with each other and both are more permissive than mlx-vlm.
Practical consequence: a request for 1000 tokens errors upstream and silently becomes
560 here.

## 5. Method, and a caution about it

Both algorithms were re-implemented from source and swept; no model was run. That is
sufficient because sizing is pure arithmetic on (width, height, budget).

**A caution worth recording.** An automated summary of `processing_gemma4.py` initially
reported that mlx-vlm *"scales down exclusively; does not enlarge images"* — which would
have been a headline divergence, since budget-fill upscaling is the entire point of 004.
Re-reading the verbatim source showed **no factor clamp and no early return**: the claim
was the summariser's inference, not the code. Prose summaries of source are not evidence
about control flow; quote the lines.

## 6. What would close the gap

1. **Golden test pinning both engines on extreme aspects.** The `x/mlxrunner`
   golden-test harness already exists. Pin ≥100:1 and ≥1:100 at budgets 70 and 140,
   where the clamp fires, and assert the MLX and llama-server grids match.
2. **Read `max_side_length`** in `processing_gemma4.py` and decide whether the fallback
   is worth porting, or whether our clamp-and-shave is strictly better (it lands on
   budget; the fallback's behaviour is unknown).
3. **Watch mlx-vlm's nemotron support.** Issue #1090 shows vision can silently no-op
   there. If this fork ever ports nemotron MLX vision, assert image tokens are non-zero
   rather than trusting the path.
4. **Track the two upstream MLX PRs — the merge surface is about to move.** Both
   [#17487](https://github.com/ollama/ollama/pull/17487) and
   [#17600](https://github.com/ollama/ollama/pull/17600) rewrite
   `x/mlxrunner/{client,pipeline,runner}.go`, which this fork's MLX vision work sits on
   top of. Syncing past either will be a genuine merge, not a fast-forward, and the two
   PRs disagree with each other on the media interface. Decide before syncing whether to
   rebase onto whichever lands or to keep diverging.
5. **Consider whether the sizing work is worth upstreaming.** #17154 raises the budget
   without fixing the geometry; ADR 0008 and `llama/compat/004` do fix it, with
   measurements. If the goal is to stop carrying the patch, that is the contribution
   with no upstream equivalent.

## Sources

- mlx-vlm — <https://github.com/Blaizzy/mlx-vlm>; gemma4 module
  <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/README.md>;
  sizing source `mlx_vlm/models/gemma4/processing_gemma4.py`
- mlx-vlm issue #1090 (nemotron vision) — <https://github.com/Blaizzy/mlx-vlm/issues/1090>
- gemma-4-swift-mlx — <https://github.com/VincentGourbin/gemma-4-swift-mlx>
- mlx-gemma4 — <https://github.com/FakeRocket543/mlx-gemma4>
- mlx-vlm-falcon — <https://github.com/korale77/mlx-vlm-falcon>
- vllm-mlx — <https://github.com/waybarrios/vllm-mlx>
- mlx-optiq — <https://pypi.org/project/mlx-optiq/>, <https://mlx-optiq.com/docs/gemma-4>
- Qwen3-VL MLX builds — <https://huggingface.co/lmstudio-community/Qwen3-VL-32B-Thinking-MLX-4bit>
- Gemma 4 model card (budget ladder) — <https://ai.google.dev/gemma/docs/core/model_card_4>
- Upstream budget-knob issue — <https://github.com/ollama/ollama/issues/15626>
- Upstream ollama, MLX vision in flight (all open as of 2026-08-07):
  - PR #17487 "mlx: add Gemma4 vision support" (@dhiltgen) — <https://github.com/ollama/ollama/pull/17487>
  - PR #17600 "mlxrunner: Add image input support" (@jessegross) — <https://github.com/ollama/ollama/pull/17600>
  - PR #17154 "llm: raise and expose Gemma 4 image token budget" (@yatishgoel) — <https://github.com/ollama/ollama/pull/17154>
  - Issue #17065 "MLX vision models do not appear to receive image input" — <https://github.com/ollama/ollama/issues/17065>
- In-repo: [ADR 0008](adr/0008-gemma4-budget-fill-restores-1120.md),
  [SPEC §2.1](spec/vision-image-token-budgets.md),
  [bbox findings](gemma4-bbox-investigation-findings.md),
  `llama/compat/004-llama-cpp-gemma4-budget-fill.patch`

## Not covered

- Only `processing_gemma4.py` and the gemma4 README were read from mlx-vlm. Its
  `vision.py` was **not** diffed against our port — this document covers **sizing
  parity only**, not encoder or projection parity.
- mlx-vlm's `max_side_length` fallback is unevaluated (§4a).
- qwen and nemotron MLX sizing were not compared at all; only gemma4 was.
- mlx-vlm moves quickly. Every quotation is from `main` as of 2026-08-07; re-check
  before relying on it.
- The three upstream PRs were read for **title, description and changed-file list
  only** — their diffs were not reviewed. Whether #17487's gemma4 sizing matches
  `BudgetFillSize`, and which media interface wins between #17487 and #17600, are both
  open questions. The "nobody upstream addresses sizing" claim rests on file lists and
  PR descriptions, not on reading #17487's 2,394 added lines.
