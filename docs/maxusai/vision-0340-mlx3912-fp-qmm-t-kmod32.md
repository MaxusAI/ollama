# SOLVED: the gemma4 26b/31b vision shift on 0.34.0 is upstream MLX #3912

**Status: ROOT CAUSE FOUND AND CONFIRMED. This is not a regression — it is a
kernel correctness FIX, and 0.34.0 is the more correct build.**

## Root cause

Upstream **MLX PR #3912** fixed a Metal kernel bug in `fp_qmm_t` that reads past
the K dimension when **`K mod 32 == 16`**. Our v0.34.1 fold (#302) crossed that
fix when the MLX pin moved `ce916dbb -> d9add9d1`.

`K = 4304` is the vision tower's `mlp.down_proj` contraction dimension in the
gemma4 **26b and 31b** nvfp4 checkpoints — stored packed as `[1152, 538]` at 8
values per word, cross-checked by the group-16 scale tensor at `[1152, 269]`.
`4304 = 134*32 + 16`.

**12b carries no nvfp4 vision `down_proj` at all** (its text ones are K = 15360
and 65536, both `mod 32 == 0`). It was never a lucky control — it is
structurally immune, which is why it came back bit-identical on every build.

Confirmed at kernel level, independent of models or scores (guard test in #315):

```
MLX ce916dbb (pre-fix)   K=4288  max 0.000152588   0/294912 elements over 1.0
                         K=4304  max 24.0494       232722/294912   <- 79% corrupted
MLX d9add9d1 (post-fix)  K=4288  max 0.000152588   0/294912
                         K=4304  max 0.137939      0/294912
```

The aligned control is bit-identical across both libraries, so the two differ
only on the one path.

### What this means for the numbers below

The tier drop is **an nvfp4 quantization cost that the broken kernel was
masking**, not a defect the fix introduced and not a defect to undo. The 9px
score of 4 was measured on a build corrupting most of that matmul's outputs.
The score of 3 is what nvfp4 costs on this sample once the arithmetic is
correct — see "The quantization control" below, where the unquantized bf16
checkpoint scores 4.

It also resolves what this document previously recorded as an unexplained
paradox — that 0.34.0 was *closer* to the mlx-vlm reference while scoring worse.
That is what a correctness fix looks like when the error it removes happened
to compensate for a quantization loss on the scored sample.

**Prior gemma4 26b/31b vision numbers measured before this fold went through a
kernel corrupting 79% of the vision tower's down_proj outputs.** Anything
compared against those baselines should be re-measured rather than trusted.

## The quantization control (2026-09-18)

**Read this table before re-measuring anything here.** Every cell below is
recorded so these arms do not have to be run again.

All three `gemma4:31b` quantizations, one binary `0.34.0-maxusai-8a7ba949`,
payload `0.32.2-61-gd9add9d` (post-#3912), `num_ctx=16384`, `num_predict=2200`
think-off / `8192` think-on, powermode 2, scored by `finetext_probe.py`:

| model | vision tower | LM | think-off | think-on |
|---|---|---|---|---|
| `31b-nvfp4` | nvfp4 | nvfp4 | `[4,4,4,3,3]` | `[4,4,4,3,3]` |
| `31b-mxfp8` | **bf16** | mxfp8 | `[4,4,4,3,2]` | `[4,4,4,4,3]` |
| `31b-mlx-bf16` | **bf16** | bf16 | `[4,4,4,4,3]` | `[4,4,4,4,3]` |

Tiers are `[22px, 16px, 12px, 9px, 7px]`. Both nvfp4 rows reproduce the
campaign cells exactly (positive control: `eval=263` think-off, tiers to the
digit), so the harness measures the same quantity today as during the campaign.

**bf16 is a clean control by construction.** `mlxrunner/model/linear.go`
dispatches on the presence of a `.weight_scale` sibling; absent it the factory
returns `nn.NewLinear`, whose `Forward` is `x.Matmul(w)` — no `QuantizedMatmul`,
no `fp_qmm_t`, no global scale. `31b-mlx-bf16` carries **zero** `.weight_scale`
tensors across 1247 (59.2 GiB).

**`31b-mxfp8` keeps `down_proj` at bf16 while quantizing the rest of the
tower to 8-bit.** Its vision L0 `mlp.down_proj` blob is 9,916,560 bytes in both
the mxfp8 and the bf16 checkpoint — byte-identical, exactly
`1152 × 4304 × 2 + 144` header, same blob digest `4b80a6f9…`. That one tensor
is shared; the other 162 vision weight layers are not. Reading only
`down_proj` and calling the tower bf16 was an error in the first version of this
section.

The same arithmetic re-derives this document's shape claim from file sizes
alone, with no code and no model load: the nvfp4 blob is
`2,789,452 = 2,479,104 (4-bit packed) + 309,888 (group-16 scales) + 460`,
giving K = 4304 and group = 16.

### What the control establishes

**The model's true 9px answer is 4** — bf16, with no quantization anywhere,
scores 4 in **10 of 10** reps. nvfp4 on the fixed kernel reaches it 2/10; nvfp4
on the broken kernel reached it 14/14. The kernel defect was compensating for an
nvfp4 quantization loss on this sample, so #3912 did not cost a tier, it stopped
hiding one.

**The cell is bimodal, and the tables above are n = 1.** Ten reps per
quantization, think-off, same binary and window:

| model | vision tower | LM | 9px per rep | rate at 4 |
|---|---|---|---|---|
| `31b-mlx-bf16` | bf16 | bf16 | `[4,4,4,4,4,4,4,4,4,4]` | **10/10** |
| `31b-mxfp8` | **bf16** | 8-bit | `[3,3,3,3,3,3,3,3,3,3]` | **0/10** |
| `31b-nvfp4` | nvfp4 | 4-bit | `[4,4,3,3,3,3,3,3,3,3]` | **2/10** |

**None of these three arms isolates the encoder from the language model.**
Reading the whole vision tower rather than one tensor:

| checkpoint | vision `down_proj` | vision `gate/up/q/k/v/o` | LM |
|---|---|---|---|
| `31b-nvfp4` | 4-bit | 4-bit | 4-bit |
| `31b-mxfp8` | **bf16** | **8-bit** | 8-bit |
| `31b-mlx-bf16` | bf16 | bf16 | bf16 |

`31b-mxfp8`'s tower is 8-bit in 162 of its 356 vision layers, with only
`down_proj` left at bf16 — it is not a bf16-tower arm. Each checkpoint varies
tower precision and LM precision together, so none of them attributes the glyph
to either.

What the reps do establish is that **the cell does not order by numerical
precision**: `31b-mxfp8` carries a bf16 `down_proj` — the very tensor #3912
corrupts — and an 8-bit tower and LM, and still scores 0/10, below the 4-bit
`31b-nvfp4`'s 2/10. A metric where more precision scores worse is not measuring
quality, and is not a readout of the vision encoder.

The single-capture tables in this section therefore report the minority mode
roughly one time in eight. They are kept because they are what was measured, but
**no conclusion should rest on one capture of this cell**, here or elsewhere in
this repo.

That is consistent with the golden-parity delta, which measures the fused
quantized matmul against dequantize-then-matmul — the arithmetic the nvfp4
weights actually encode — and improves `0.1406 → 0.0898` with the fix, landing
exactly on MLX-CUDA's `0.0898` (#316).

### The confound, stated plainly

`31b-mxfp8` carries bf16's exact encoder and still drops to 3 at think-off. So
**language-model quantization alone moves the 9px tier**, and because nvfp4
quantizes both the tower and the LM, these three checkpoints cannot pin nvfp4's
3 to the vision tower specifically. Isolating that needs an nvfp4-vision +
bf16-LM checkpoint, which the store does not contain.

The general point outlives this investigation: a recall tier is not a clean
readout of vision-encoder correctness, and a tier move across a build bump is
not evidence about a kernel until the unquantized arm has been measured on the
same binary and window.

### Tier history, so it is not re-measured

`gemma4:31b-nvfp4`, all builds, both think modes — transcribed from the
`ft_*.json` capture files by script (the suite gitignores them, so this table is
their durable form):

| build | MLX pin | think-off | think-on | runs |
|---|---|---|---|---|
| `0.33.0-maxusai-21cfe88e` | pre-fold | `[4,4,4,4,3]` | `[4,4,4,4,3]` | 2 |
| `0.33.2-maxusai-2b95b4a5` | `c793734e` | `[4,4,4,4,3]` | `[4,4,4,4,3]` | 5 |
| `0.34.0-maxusai-907deffd` | `ce916dbb` | — | `[4,4,4,4,3]` | 3 |
| `0.34.0-maxusai-8a7ba949` | `d9add9d1` | `[4,4,4,3,3]` | `[4,4,4,3,3]` | 5 |

`26b-nvfp4` and `12b-nvfp4` are **unchanged across all three builds**
(`[4,4,4,3,3]` and `[4,4,3,0,0]` think-off) — the 26b's encoder moved from the
same fix while its tier did not, which is the first sign the tier and the
encoder are not the same measurement.

### One caveat on build-pair attribution

`MLX_VERSION` did not move `ce916dbb → d9add9d1` along one line. `fbedf5066`
moved the fold `37c26e57 → ce916dbb`, while upstream went
`37c26e57 → cbb4059d → d9add9d1` separately — so the two builds sit on
**different MLX branches**, not consecutive commits. #3912 itself survives that,
because the kernel reproducer is selected by K (clean at `K mod 32 == 0`, broken
only at 16) rather than by build pairing. No purely build-pair attribution
does.

### Two things that made this expensive, worth carrying forward

**The symptom was almost invisible.** Not a crash, not an obviously wrong
answer: one fine-text OCR tier, one model, one think mode — and the model stayed
coherent throughout.

**Nobody read upstream's PR history.** The whole investigation below bisected our
own fold and treated the MLX pin as an opaque range. The answer was in
MLX's changelog the entire time. The sibling ROCm investigation reached its own
root cause the same way, on the same day.

## The finding

`gemma4:31b-nvfp4`, think **on**, `finetext` arm, `num_ctx=16384`
`num_predict=8192`. The 9px exact-match recall tier drops from 4 to 3 between
`0.33.2-maxusai-2b95b4a5` and `0.34.0-maxusai-8a7ba949`.

Twelve runs, three per build per grammar condition:

```
0.34.0-maxusai-8a7ba949  constrained    9px = [3, 3, 3]
0.34.0-maxusai-8a7ba949  unconstrained  9px = [3, 3, 3]
0.33.2-maxusai-2b95b4a5  constrained    9px = [4, 4, 4]
0.33.2-maxusai-2b95b4a5  unconstrained  9px = [4, 4, 4]
```

Six more under forced greedy decoding (`temperature 0`, `top_k 1`):

```
0.34.0-maxusai-8a7ba949  GREEDY think-on  9px = [3, 3, 3]
0.33.2-maxusai-2b95b4a5  GREEDY think-on  9px = [4, 4, 4]
```

18 runs, no overlap. This is not sampling luck.

## What is NOT affected

**Think-off.** The full campaign scored all five nvfp4 models think-off on both
builds: every OCR tier, every box/label/colour count, every invoice field and
every multi-image verdict identical, answer-token counts identical. Only scene
IoU moved, in the third decimal, and upward on four of five. The 31b think-off
row held `4,4,4,4,3` on both builds — the same arm that moves under think-on.

Think-off also runs under `format:"json"` (the default in `client.generate`), so
constrained sampling is exercised there too and found inert.

## Four falsified hypotheses

Recorded because they are the four explanations anyone reaches for first, and
each cost a run to kill.

**1. #301's whitespace bound.** `max_whitespace_cnt` is present on 0.34.0 and
absent on 0.33.2, and acts only on constrained output — so it looked like the
delta. Falsified: removing the grammar entirely changes nothing on either build
(rows 2 and 4 above). #301 is exonerated.

**2. Reasoning length.** 0.34.0 terminated its reasoning earlier on the first
samples seen (1821 vs 2046 thinking chars), which suggested truncated reasoning
costing fine-tier recall. Falsified by the full set: 0.33.2 produced the
shortest run of all at **1545** thinking chars and still scored 4, while 0.34.0
thought **2423** and still scored 3. Thinking length does not predict the
outcome across a 1.6x range.

**3. The resize kernel.** A `b9888..b10864` diff of the gemma4 branch shows
`image_resize_algo` moving to `RESIZE_ALGO_BICUBIC` alongside the token-limit
change, which reads as a simultaneous b10864 change. It is not: b10630 — the
payload behind 0.33.2 — already sets BICUBIC, so the kernel is identical on both
sides. The MLX path has always used CatmullRom (Keys a=-0.5, the bicubic family)
regardless. See `llama/compat/README.md`.

**4. The sampling path.** Think-off is greedy and unaffected; think-on is
non-greedy and moved, so sampling was the sharpest remaining asymmetry.
Falsified: the gap persists unchanged under forced greedy.

## Corrected: greedy spread exists on BOTH builds; drafting amplifies it

An earlier revision of this document claimed 0.34.0 is nondeterministic under
greedy while 0.33.2 is deterministic, and offered that as the better bisect
target. **That claim was a three-sample artefact and is withdrawn.** It is kept
here rather than deleted, because it is the same small-sample error this
repository has made before and the correction is the useful part.

Pooling every greedy run (`temperature 0`, `top_k 1`, think-on, same prompt):

| build / config | runs | identical |
|---|---|---|
| `0.33.2-maxusai-2b95b4a5` | 6 | 5x `eval=1588 think=2046`, 1x `1584/2038` |
| `0.34.0-maxusai-8a7ba949`, drafting **off** | 3 | 2x `1598/2423`, 1x `1591/2046` |
| `0.34.0-maxusai-8a7ba949`, drafting **on** | 3 | 0 — three distinct |

What this supports:

- **MLX greedy decoding has baseline run-to-run spread on both builds.** The
  fork's own comment at `mlxrunner/speculate.go` already said so — "within the
  run-to-run spread MLX already has".
- **Drafting under a grammar amplifies it substantially**: zero identical runs
  out of three with it on, two of three with it off. `draftUnderGrammar` does
  not exist on 0.33.2 at all; it arrived with `068a98cd0`, which adopted
  upstream's default and put it behind `OLLAMA_MLX_DRAFT_UNDER_GRAMMAR`. That
  knob is read once at process init, so it is a server-start setting, not a
  per-request one.

What this does NOT support: that 0.34.0 introduced nondeterminism. The first
three 0.33.2 runs came up identical and a headline was built on them; a later
run in the identical configuration produced an outlier.

**The tier drop is unaffected by any of this.** 9px is 3 on 0.34.0 and 4 on
0.33.2 in twelve runs each — constrained, unconstrained, greedy, and greedy with
drafting disabled. 24 runs, perfect separation, no configuration moves it. That
is the finding to bisect; the determinism question is a separate and much softer
observation.

## Bisect: everything is in #302

Bisected on the ENCODER rather than the tier. The tier needs a scored benchmark
and sits on a quantity that drifts hundreds of tokens between runs; the encoder
measurement is deterministic and answers one question per build in seconds.

Midpoint `907deffd` — the v0.34.0 fold plus #300 and #301, pins `b10760` /
MLX `ce916dbb`, distinct from both endpoints:

| | 26b encoder maxΔ | think-on 9px | greedy eval range |
|---|---|---|---|
| `2b95b4a5` (0.33.2) | 0.2266 | 4 | 1584-1588 (4) |
| `907deffd` (midpoint) | **0.2266** | **4** | 1289-1290 (**1**) |
| `8a7ba949` (v0.34.1 fold) | 0.0508 | 3 | 1290-2105 (**815**) |

**Three independent observables flip at the same commit boundary** — image
embeddings, OCR tier, and run-to-run spread. All clean on the midpoint, all
moving in #302. 26b and 31b agree, and the 12b control is bit-identical on all
three builds.

That eliminates the entire v0.34.0 upstream fold, #300, and #301. It does not
prove one cause, but it rules out the findings being scattered across the fold
and needing separate hunts.

**Remaining suspects, all inside #302:** MLX `ce916dbb -> d9add9d1` (24 upstream
commits, prime suspect since MLX is the vision compute backend), MLX-C
`c74db530 -> ebc88f10`, the scoped-array-lifetimes refactor, the NVFP4
global-scale / `gather_qmm` work, and the `ec3cc2307` KV-buffer cherry-pick.

## A separation route that does not work

The MLX pin and the Go changes ship together, so no bisect separates them.
Pairing them deliberately — `8a7ba949`'s Go with `ce916dbb`'s MLX payload —
IS NOT POSSIBLE:

```
CHECK failed: mlx_stream_thread_local_new_
SIGSEGV during cgo execution
mlx._Cfunc_mlx_install_capture_handler()
mlx.init.1()  at mlx/mlx.go:43
```

The new Go calls `mlx_install_capture_handler` at package init and `ce916dbb`'s
MLX-C lacks the symbol it needs, so it dies before any test runs. Separating the
halves requires reverting Go changes against the new payload, not swapping
payloads. Worth knowing before spending time on the cheap-looking approach.

It also means payload/binary skew fails LOUDLY on this path rather than
silently, which is stronger than what `mlx_payload_pin` was designed to catch.

## Next step not taken

Splitting #302 further, or the MLX-CUDA comparison in #312 — which is now the
cheapest remaining cut, because the surviving suspects divide into MLX Metal
kernels (which CUDA would not reproduce) and shared Go/MLX-C layers (which it
would).

## Reproducing

Both pairings are archived, binary AND payload, so neither needs a rebuild:

```
~/.ollama/binaries/ollama-0.33.2-maxusai-2b95b4a5  + payload-0.33.2-maxusai-2b95b4a5/
~/.ollama/binaries/ollama-0.34.0-maxusai-8a7ba949  + payload-0.34.0-maxusai-8a7ba949/
```

Assemble each as a self-contained install (`<dir>/ollama` plus `<dir>/lib/ollama`);
`libOllamaRoots()` checks `<exeDir>/lib/ollama` first on darwin, so it resolves
its own payload and never consults the fork checkout.

**Assert the served version before measuring.** `serve-apple-mlx.sh` kills only
its own `$BIN`, so switching builds leaves the previous one holding `:11436`;
the new server fails to bind, the readiness curl is answered by the process that
was supposed to be replaced, and the run measures the wrong build under the right
label. Free the port by PORT, then check `/api/version` matches before running.
