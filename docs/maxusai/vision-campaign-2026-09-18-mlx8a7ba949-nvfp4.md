# Vision campaign 2026-09-18 — the nvfp4 fleet on the 0.34.0 fold, after the MLX #3912 kernel fix

One host, one build. Apple Silicon `10.8.0.3`, native serve on `:11436`
(`serve-apple-mlx.sh`, cold server per cell, `OLLAMA_MAX_LOADED_MODELS=1`),
build **`0.34.0-maxusai-8a7ba949`** — llama.cpp `b10864`, MLX `d9add9d1`.
Campaign tag `mlx8a7ba9v2nv1_`, driven from merged main `2bea4b56c`.
**Every cell ran at `powermode=2`.** Wall clock 21:47:04 → 22:44:51, both
phases `rc=0`.

This repeats [the 2026-08-31 campaign](vision-campaign-2026-08-31-mlx0332-nvfp4.md)
on the next fold, so the two are directly comparable: same host, same suite,
same sampling, same power mode.

**The question.** This is the first full campaign after upstream MLX
[#3912](https://github.com/ml-explore/mlx/pull/3912) fixed `fp_qmm_t`'s past-K
read at `K mod 32 == 16` — the gemma4 vision tower's `mlp.down_proj` contraction
dimension (K = 4304) in the 26b and 31b nvfp4 checkpoints. **Every Metal 26b/31b
vision number taken before `8a7ba949` went through the corrupted kernel** and is
superseded by this run
([vision-0340-mlx3912-fp-qmm-t-kmod32.md](vision-0340-mlx3912-fp-qmm-t-kmod32.md),
[#312](https://github.com/MaxusAI/ollama/issues/312)).

**Scope.** Think-off runs all five nvfp4 tags. Think-on runs only
`gemma4:31b-nvfp4` and `qwen3.8:27b-nvfp4`, matching the 0.33.2 campaign's
narrowing so the two remain comparable; the repo convention is to run both modes,
so this is a stated cost decision rather than an omission.

## The request this campaign sends

`emit_request.py gemma4:31b-nvfp4 false` — pasted verbatim (SPEC H7/H9; the
campaign drove `http://127.0.0.1:11436`, the `HOST:11497` placeholder below is
the emitter's):

```
POST http://HOST:11497/api/chat
{
  "model": "gemma4:31b-nvfp4",
  "stream": false,
  "options": {
    "num_predict": 2200,
    "num_ctx": 16384,
    "temperature": 0
  },
  "format": "json",
  "think": false,
  "messages": [
    {
      "role": "user",
      "content": "You are a localization service. Find every distinct\ncoloured shape in this image and report where each one is.\n\nUse \"bbox_type\": \"norm1000\" — each axis scaled independently to 0-1000, x by\n1000/width and y by 1000/height. The coordinate space is 1000x1000 whatever the\nimage's shape is.\n\n\nGive each coordinate its own named field: \"x1\", \"y1\", \"x2\", \"y2\". Do not use a\npositional array, and do not declare a \"coord_order\" — named fields state their own order.\n\nDeclare the convention on EVERY object, next to that object's coordinates.\n\nEach box covers the shape itself, not its label text. A box that disagrees with\nits declaration is worse than no answer at all, because a consumer trusts the\ndeclaration.\n\nThe FIRST entry must be a calibration entry with label \"__IMAGE__\" whose\ncoordinates cover the ENTIRE image, corner to corner, in the same convention as\neverything else. Then list the shapes.\n\nRespond with a SINGLE JSON object, no prose:\n{\n  \"objects\": [{\"label\": \"__IMAGE__\", \"bbox_type\": \"norm1000\", \"x1\": , \"y1\": , \"x2\": , \"y2\": },\n              {\"label\": \"<uppercase code word above the shape>\",\n               \"bbox_type\": \"norm1000\", \"x1\": , \"y1\": , \"x2\": , \"y2\": }]\n}",
      "images": [
        "<base64 of visimgs/scene_hd.png>"
      ]
    }
  ]
}
```

## 1. Think-off, five nvfp4 models

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | **0.955** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | **0.972** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.961** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **0.999** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | **0.961** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Answer tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:12b-nvfp4 | **MLX** | 16384 | 4 | 4 | 3 | 0 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 535 | 111 | 3693 | 5.3 | 682 |
| gemma4:26b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 538 | 144 | 9175 | 3.9 | 919 |
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 3 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 537 | 50 | 1320 | 12.0 | 300 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 67 | 2783 | 9.1 | 394 |
| qwen3.6:35b-a3b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 549 | 118 | 15069 | 4.8 | 743 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.34.0-maxusai-8a7ba949 · think=false

## 2. Think-on, the two cells that terminate

## Scene grounding (six objects, norm-1000 boxes) + document extraction

| Model | Engine | num_ctx | Scene bbox IoU | Boxes / labels / colors | Serial | Invoice (items · qty+price · total) | name_bbox in-band |
|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | **0.966** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 4/5 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | **1.000** | 6/6 · 6/6 · 6/6 | ✅ | 5/5 · 5/5 · ✅ | 5/5 |

## Fine-text OCR (exact-match recall per size tier, /4) + multi-image + throughput

| Model | Engine | num_ctx | 22px | 16px | 12px | 9px | 7px | Multi-image (3 imgs) | Multi anchored | Think tok | Gen tok | Gen tok/s | Prefill tok/s | s/req | req/h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma4:31b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 3 | 2 | ✅ q1 + q2 + q4-bbox | ✅ q1 + q2 + q4-bbox | — | 5615 | 45 | 150 | 134.8 | 27 |
| qwen3.8:27b-nvfp4 | **MLX** | 16384 | 4 | 4 | 4 | 2 | 0 | ❌ q4_bbox_hit | ✅ q1 + q2 + q4-bbox | — | 1122 | 47 | 1990 | 25.2 | 143 |

Provenance (from score files): host(s) http://127.0.0.1:11436 · build(s) 0.34.0-maxusai-8a7ba949 · think=on

## 3. What moved against 0.33.2 — one cell, and it is a tripwire

Across three gemma4 models x five size tiers x both finetext arms x three
builds (`0.33.0`, `0.33.2`, `0.34.0`), **exactly one cell differs**:

| model | 0.33.0 | 0.33.2 | 0.34.0 |
|---|---|---|---|
| `12b-nvfp4` think-off | `[4,4,3,0,0]` | `[4,4,3,0,0]` | `[4,4,3,0,0]` |
| `26b-nvfp4` think-off | `[4,4,4,3,3]` | `[4,4,4,3,3]` | `[4,4,4,3,3]` |
| `31b-nvfp4` think-off | `[4,4,4,`**4**`,3]` | `[4,4,4,`**4**`,3]` | `[4,4,4,`**3**`,3]` |

Everything else — every other tier, both other models, the grounding and
document arms — is unchanged.

### The cell is one character, and it is not deterministic

The suite runs finetext **twice** per cell: `vision_suite.py`'s folded arm
(`scores_<tag>.json`) and `finetext_probe.py` (`ft_<tag>.json`), which are
independent generations. On `0.34.0` the two arms disagreed with byte-identical
provenance — same `prompt_sha` `2609dc6cba954ae7`, same `images_sha`
`a1d6d5519519ecb4`, same `16384/2200` window, same `eval_count=263`, same
`answer_chars=351`, `powermode=2`:

```
A (scored 9px=4)  ... "HVA-4828-EN89", "RNK-0391-DW18", "JFR-2880-PW01", ...
B (scored 9px=3)  ... "HVA-4828-EN89", "RMK-0391-DW18", "JFR-2880-PW01", ...
```

One character, `N` against `M`. Ground truth (`visimgs/finetext_gt.json`) is
**`RNK-0391-DW18`**, so A is right and B is wrong — the model misreads N as M,
the hardest confusion in that glyph set at 9px.

### What the reps show, and what they cannot

Ten reps per quantization, think-off, one binary (`8a7ba949`, payload
`d9add9d1`), same window, `powermode=2`:

| model | vision tower | LM | 9px per rep | rate at 4 |
|---|---|---|---|---|
| `31b-mlx-bf16` | bf16 | bf16 | `[4,4,4,4,4,4,4,4,4,4]` | **10/10** |
| `31b-mxfp8` | **bf16** | 8-bit | `[3,3,3,3,3,3,3,3,3,3]` | **0/10** |
| `31b-nvfp4` | nvfp4 | 4-bit | `[4,4,3,3,3,3,3,3,3,3]` | **2/10** |

**None of these three arms isolates the encoder from the language model.**
Reading the whole tower rather than one tensor:

| checkpoint | vision `down_proj` | vision `gate/up/q/k/v/o` | LM |
|---|---|---|---|
| `31b-nvfp4` | 4-bit | 4-bit | 4-bit |
| `31b-mxfp8` | **bf16** | **8-bit** | 8-bit |
| `31b-mlx-bf16` | bf16 | bf16 | bf16 |

`31b-mxfp8` shares bf16's `down_proj` blob exactly (9,916,560 bytes, digest
`4b80a6f9…`) but 162 of its 356 vision layers are 8-bit. Each checkpoint moves
tower and LM precision together, so none of them attributes the glyph to either.

What the reps do establish is that **the cell does not order by numerical
precision.** `31b-mxfp8` carries a bf16 `down_proj` — the exact tensor #3912
corrupts — plus an 8-bit tower and LM, and scores 0/10, *below* the 4-bit
`31b-nvfp4`'s 2/10. A metric on which more precision scores worse is not
measuring quality, and cannot be read as a verdict on the encoder.

The isolating arm does not exist in this store: it would be an nvfp4 tower with
a bf16 LM, or the converse. The MLX-CUDA session is pulling the registry's
current `31b-nvfp4`, which carries a **bf16 tower with a 4-bit LM** — against
our local copy's 4-bit tower and 4-bit LM, that varies the tower alone.

### The local 31b checkpoints are preserved under tower-qualified tags

The MLX-CUDA session found that our local `gemma4:31b-nvfp4` and the registry's
current tag **share a config digest while differing in 194 vision layers** —
ours has a 4-bit tower, the registry now ships that tag with a bf16 one. Pulling
the registry tag would therefore overwrite the checkpoint every number in this
document was measured on, under the same name, with no version change to warn
anyone.

The three 31b MLX checkpoints are copied to tower-qualified tags before any pull
(manifests only — blobs are shared by digest, so the copies cost no disk):

| preserved tag | vision `down_proj` | vision `gate/up/q/k/v/o` | LM |
|---|---|---|---|
| `gemma4:31b-nvfp4-tower-nvfp4` | 4-bit | 4-bit | 4-bit |
| `gemma4:31b-mxfp8-tower-mxfp8` | **bf16** | 8-bit | 8-bit |
| `gemma4:31b-mlx-bf16-tower-bf16` | bf16 | bf16 | bf16 |

**The `-tower-mxfp8` name is lossy on purpose** and this table is the authority:
that tower is 8-bit in 162 of its 356 vision layers with `down_proj` alone left
at bf16 — which is why that checkpoint never meets the #3912 defect even on a
pre-fix build.

`gemma4:12b-nvfp4` needs no such tag: it has no `vision_tower.encoder` at all,
only a `vision_embedder` (patch dense plus positional), which is the structural
reason it was immune to #3912 throughout.

### The 26b is the counter-control

The 26b's encoder was corrupted **more** than the 31b's by the same kernel bug —
max sampled element delta `0.2266 -> 0.0508` across the fix, against the 31b's
`0.1406 -> 0.1094` ([upstream-sync-0.34.1.md](tasks/upstream-sync-0.34.1.md)
§ "The MLX pin move and the vision encoder"). If corrupted vision embeddings
systematically helped fine-text OCR, the far-more-corrupted model should show
the larger score effect. **It shows none at all**, on any tier, on any build.

### Counting both arms, for the record

| build | 9px = 4 | 9px = 3 |
|---|---|---|
| `0.33.0` / `0.33.2` (pre-#3912 kernel) | **14** | 0 |
| `0.34.0` (post-#3912 kernel), campaign captures | **1** | **15** |
| `0.34.0`, N=10 reps above | **2** | **8** |

A single observation of this cell reports the minority mode roughly one time in
eight. **Any conclusion drawn from one capture of the 9px tier is unsound**, and
that includes conclusions drawn in this repo before today.

## 4. Limits

- **`qwen3.8:27b-nvfp4` think-on fails the free multi-image bbox arm**
  (`❌ q4_bbox_hit`) while passing the anchored one. Unrelated to the gemma4
  kernel question; flagged here as its own regression candidate, not
  investigated in this campaign.
- **Think-token split is absent** from the phase 2 table (`Think tok —`);
  `token_split.py` has not been run over these captures, so the think/answer
  split is char-based only.
- Think-on covers two models, not five (see Scope).
- `n = 1` per cell, which is the suite's standing convention for quality
  (quality is measured at `n = 1`; a throughput anomaly is what triggers a
  targeted re-run). The bimodality above is the
  first measured case where that convention can report the wrong mode, and it
  is the reason the follow-up runs `N = 10`.

## 5. Settled — on real OCR the kernel defect is undetectable

The 9px cell could not answer whether the pre-#3912 kernel reads fine text
better, so the question was put to **OCRBench v1** (`echo840/OCRBench`,
contains-match, lmms-eval semantics via `extbench.py`), all **1000** items,
`gemma4:31b-nvfp4` think-off, `num_ctx=16384`, on both archived builds — the
same local checkpoint (4-bit tower, 4-bit LM) under each binary, so only the
build differs:

```
0.34.0-maxusai-8a7ba949  (MLX d9add9d1, FIXED)   835/1000 = 0.8350
0.33.2-maxusai-2b95b4a5  (MLX c793734e, BROKEN)  833/1000 = 0.8330

both right 826 | both wrong 158 | FIXED only 9 | BROKEN only 7
discordant 16   exact McNemar two-tailed p = 0.8036   -> not significant
churn 16/1000 = 1.6%
```

**The two builds are statistically indistinguishable.** Two items separate them
in 1000. The kernel defect — which corrupts 232722 of 294912 elements at
K = 4304, M = 256 in isolation — is not detectable in end-to-end OCR accuracy.

### The 200-item slice was misleading, in both directions

An earlier 200-item run gave `FIXED only 4 / BROKEN only 0` and was reported here
as directionally unanimous but underpowered. Extending to 1000 shows the
direction was noise: across the remaining 800 items the split is 5 fixed / 7
broken. A unanimous 4–0 cannot reach significance in a two-tailed exact test —
that caveat was right, and the extension is why it mattered.

The slice was also unrepresentative of the benchmark. Per-chunk accuracy on the
fixed build runs `0.875, 0.845, 0.945, 0.845, 0.665` — the last 200 items are
much harder. **Reading the first 200 as 87.5% and comparing it to the tech
report's 88.3 reasoning-off figure was wrong**; the full-benchmark number for
this checkpoint is 83.5%, and the gap to 88.3 is most plausibly nvfp4
quantization, though that has not been measured here against an unquantized run.

### What the 16 discordant items look like

Descriptive only — 9 against 7 is not a difference, and the classification below
is a reading of the outputs, not a tested claim.

| item | gold | fixed | broken |
|---|---|---|---|
| 0 | `CENTRE` | `Centre` | `Centuries` |
| 78 | `DAVIDSON` | `Davidson` | `Davison` |
| 93 | `CORONAD` | `CORONADO` | `CORONA` |
| 218 | `100972` | `100972` | `10972` |
| 374 | `72.7` | `72.7%` | `12.7%` |
| 239 | `27299` | `272.99` | `27299` |
| 265 | `TISPPIP` | `TISPP IP` | `TISPPIP` |
| 348 | `TRANSAVIA.COM` | `Transavia` | `transavia.com` |
| 931 | `AO = OC = OB = OD` | `OB - OD` | `OB = OD` |

The fixed build's wins are character reads (`Davison`, `CORONA`, `10972`,
`12.7%` are dropped or substituted characters). Several of the broken build's
wins are punctuation and spacing under a strict contains-match scorer
(`272.99` for `27299`, `TISPP IP` for `TISPPIP`, `Transavia` truncating the
domain) rather than misreads. Only two — item 670 `England` for `australia` and
item 673 `Thursday` for `monday` — are outright errors by the fixed build.

### The decision this answers

**Do not revert the kernel.** Not because reverting would cost OCR accuracy —
it demonstrably would not, at this precision — but because the fixed kernel is
arithmetically correct, matches MLX-CUDA's golden delta exactly at `0.0898`, and
the 9px tier that motivated the question was never measuring the encoder. There
is no accuracy argument on either side; there is a correctness argument on one.

## 6. Open

- **Which checkpoint the MLX-CUDA host measured.** Its 200-item OCRBench figure
  (0.860) is comparable to Metal's only if it ran a 4-bit tower; the registry's
  current `31b-nvfp4` ships a bf16 one, and the two differ in 194 layers.
- **No unquantized OCRBench baseline.** `31b-mlx-bf16` has not been run over the
  1000 items, so the 83.5 → 88.3 gap is attributed to quantization by plausibility
  rather than measurement.
