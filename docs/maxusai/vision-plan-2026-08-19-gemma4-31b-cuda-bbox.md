# Plan: gemma4:31b-it-q4_K_M on CUDA — trustworthy `box_2d` under thinking

**Goal.** Settle production settings for bounding boxes on **unknown images**:
a `box_2d` a consumer can trust, JSON that parses or can be salvaged, ideally
with `think=on`. Capture latency, the `num_ctx` rung reached, and prompt /
thinking / answer token counts per cell.

**Target.** `gemma4:31b-it-q4_K_M` (18.5 GB, **GGUF** Q4_K_M) on
`10.8.0.6:11497`, `0.32.14-rc0-dynres-0-ga5d6590` — a **CUDA Blackwell** host
running an **experimental mlx-cuda build that is not stable**.

**GGUF only. This is a hard constraint, not a preference.** The tag chooses the
runtime: `q4_K_M` / `q8_0` are GGUF and route to llama.cpp/CUDA; `nvfp4`,
`mxfp8` and `mlx-bf16` route to MLX. So `gemma4:31b-nvfp4` — present on the host
and superficially the "native" choice, since NVFP4 is NVIDIA's own FP4 format
for Blackwell — is precisely the tag that lands in the unstable path. Running it
would produce failures that look like model results.

Excluded for this campaign: `gemma4:31b-nvfp4`, and every `mxfp8` / `mlx-bf16`
tag. The MLX code path in this build is simply not exercised by GGUF tags, which
is what makes the host usable at all.

**Status: not started.** This is the plan, not results.

---

## 0. The thing that decides the outcome, before any run

You listed three axes — thinking, anchor, image size. There is a fourth, and it
is the one that determines whether `box_2d` is usable at all.

**`box_2d` is positional, and gemma's native order is `yxyx`** — `[y1, x1, y2,
x2]`, as `vision_suite.py:359` already encodes. SPEC
[C2](spec/vision-bbox-response-contract.md) measured the consequence: a
positional array came back `yxyx` **while declaring `xyxy`** in **11 of 26**
gemma4 cells; named `x1`/`y1`/`x2`/`y2` fields did it in **0 of 13**.

**The anchor cannot rescue this.** A transposed box in a normalized space has
the same range, the same extent aspect and no scale error — and the `__IMAGE__`
anchor is a full-image box, which is *symmetric under transposition*. It is
identical whether the model means `xyxy` or `yxyx`. So `anchor=on` does nothing
for axis order, and no numeric check downstream can detect it either. This is
the one error class that is invisible without ground truth.

**But there is real reason to expect 31b is fine.** Every measured flip was a
gemma4 **26b** variant (a4b-MoE, nvfp4, mxfp8, mlx-bf16). **31b sat in the clean
21/21 group.** So the hypothesis is that `box_2d` is safe on this specific
model — and the point of the screening stage is to establish that with evidence
rather than inherit it.

**Therefore the design carries a 4th axis: coordinate format**, `box_2d`
(positional) against named `x1..y2`. Named fields are the control that makes the
error unrepresentable; `box_2d` is the thing you want to ship. If they agree
across every geometry, `box_2d` is cleared for this model. If they diverge
anywhere, ship named fields and translate to `box_2d` at your own boundary,
where you control the order.

---

## 1. Preconditions (do these first; each is a code or baseline change)

**P0 — Preflight the host.** The platform is known (CUDA Blackwell), but
`0.32.14-rc0-dynres` is a `(platform, version)` combination
`expectations.toml` has never seen, so per
[ADR 0011](adr/0011-preflight-expectations-are-versioned-code.md) expect
**exit 4 / NEEDS_BASELINE**. Baseline against a **`cuda-dynres-*` profile, not
`mlx-cuda`** — the MLX profile is carried in `expectations.toml` as unmeasured
with reasons, and GGUF tags do not exercise that path at all. Do not start the
sweep until a baseline is recorded.

> Also worth resolving here: the build reports git `a5d6590`, which is a prefix
> of `a5d65906` — the commit recorded in the SPEC's provenance line for
> **0.32.5**-maxusai. Either the version string comes from a build argument
> rather than the tree, or a stale describe is baked into the image. Until that
> is known, the version string does not identify the source.

**P1 — Split thinking from answer tokens, exactly.** `vision_suite.py:130`
captures `message.thinking`, but `eval_count` is the total and nothing records
the split. Tokenize the two strings and record **three buckets that sum to
`eval_count` exactly**:

| field | how |
|---|---|
| `thinking_tokens` | tokenize `message.thinking` |
| `answer_tokens` | tokenize `message.content` |
| `control_tokens` | `eval_count − thinking_tokens − answer_tokens` |

`control_tokens` is the think delimiters, EOS, and the `format:json` grammar
tokens — real generated tokens that emit no visible text. **Report them; never
distribute them.** A char-proportional estimate would silently push them into
the other two buckets and misreport both.

Measured on this host, that residue is not negligible: a 62-token qwen3.8
think-on response streamed **49** chunks carrying visible text. 13 tokens — 21% —
produced no delta at all. Counting stream chunks is therefore *not* an exact
method and must not be used as one.

**Getting the tokenizer**, in preference order:

1. **Parse the GGUF on the host.** `tokenizer.ggml.tokens` / `.merges` live in
   the blob, and this is by definition the vocab the server is using. Needs file
   access on 10.8.0.6 plus the `gguf` package.
2. **HuggingFace tokenizer for the matching base model**, via `transformers` or
   the lighter `tokenizers`. `/api/show` names the tokenizer even though it nulls
   the arrays — for gemma4 read `tokenizer.ggml.model` and `tokenizer.ggml.pre`
   off the target model and pick the matching repo. Risk: a fork's conversion can
   diverge from the public repo, and gemma repos are gated.
3. **No split.** Record `eval_count`, `thinking_chars`, `answer_chars` and stop.

**Acceptance gate before any split is published:** `control_tokens` must be
**non-negative and small** — single digits to low tens — across a sample of at
least 20 responses. A wrong vocab makes it go negative or absurd, so the
tokenizer validates itself against `eval_count` and needs no separate trust
argument. If the gate fails, fall back to option 3 rather than publishing a
split that looks exact and is not.

**P2 — Add a JSON salvage pass. DONE, and the design above was wrong.**
Implemented in `salvage.py`; records `json_salvaged` and `salvage_method`
alongside the existing `json_valid`.

This section originally specified fence-stripping, quote-unescaping and
brace-repair for "malformed JSON". **The response was not malformed.** Inspecting
the actual qwen3.6 q8_0 think-on cell: `json.loads` succeeds. The model had
serialised its `answers` object into a **string inside the images array** —

```
{"images": [ {...}, {...}, {...},
             "answers\": {\"q1\": 2, ... }}}```json{  " ]}
```

— so the JSON is valid, the content is correct, and `answers` is simply not a
key. None of the four planned steps would have touched it.

The fix is `require_key`: a scorer names the key it needs, and salvage searches
string values for a `"<key>": {...}` fragment that landed in the wrong slot.
**Valid JSON is not the same as usable JSON**, and without naming the key a
scorer cannot tell "the model got it wrong" from "the model put it in the wrong
place" — which must not share a rate.

Verified against the real cell: recovered `q1: 2`, `q2: Q4/128`, matching the
same model's *passing* arm exactly. Rescoring the 18-cell qwen3.6 set flips
exactly **3 cells** ❌❌❌ → ✅✅✅, all `q8_0` think-on anchored; every other cell
is byte-stable, so this is surgical rather than a scorer made permissive.
Truncation repair and largest-object extraction are implemented too, as later
fallbacks, and `salvage_method` distinguishes them — `embedded_key` is a model
that answered correctly in the wrong slot, `largest_object` is one that barely
returned a response, and pooling those would hide the distinction.

**P3 — Never edit `vision_suite.py` while a sweep runs against it.** A live edit
cost a cell in the geometry sweep (`NameError` mid-import). Land P1 and P2,
verify with `test_rescore.py`, then start.

**P4 — Idle host.** Latency is a deliverable here, and throughput measured under
contention is not comparable. Confirm nothing else is on 11497 before starting,
and record `powermode` per cell as usual. Note 11497 is one of several servers on
10.8.0.6 (11434, 11437, 11449, 11452 also answer) — if any of those share the
GPU, "idle" means idle across all of them, not just this port.

**P5 — Separate build failures from model failures.** The build is experimental.
A transport error, a 500 or a crash is **not** a model result and must never be
scored as one. `vision_suite.py` already records `results[name] = {"error": ...}`
distinctly from a score, so the requirement is analytical: report error cells as
their own count, never fold them into a ❌ rate, and re-run any cell that errors
before drawing a conclusion from it. If error cells exceed roughly one in twenty
even on the GGUF path, stop — the finding is about the build, not the model, and
the campaign cannot say anything about gemma4 until it is stable.

---

## 2. The matrix

**Geometries** — 7 of the 14, chosen so each still isolates something:

| geometry | W×H | why it stays in |
|---|---|---|
| `hd` | 1920×1080 | control; ties to the whole existing corpus |
| `sq320` | 320×320 | below token floor; square — both C7 checks dead at once |
| `vga` | 800×600 | 4:3, `max(W,H) ≤ 1000` — C7 range check ambiguous |
| `portrait` | 1080×1920 | aspect isolator; same pixel count as `hd` |
| `uhd4k` | 3840×2160 | frame < input (C17) |
| `paste3` | 1235×1181 | near-square — C7 aspect check dead, and *large* |
| `paste1` | 1668×733 | extreme 2.276 aspect; transposition most visible here |

`paste1` earns its place twice over: at aspect 2.276 an `xyxy`/`yxyx` confusion
produces boxes that are wildly wrong rather than subtly wrong, so it is the
geometry where an order flip is easiest to see against ground truth.

**Factors** — 2 think × 2 anchor × 2 format = 8 configurations.

### Stage A — screening, `think=off`, n=1

Think is fixed off here, so the live factors are 2 anchor × 2 format = **4
configurations** × 7 geometries = **28 cells**. Deterministic at temperature 0,
so n=1 is sufficient. Answers, cheaply:

- does `box_2d` ever come back `yxyx` on **31b**? (the C2 question)
- does the anchor change anything when the format is already named?
- does any geometry break what `hd` says?

**Gate:** if `box_2d` order is clean at all 7 geometries, carry both formats into
Stage B. If it flips anywhere, drop `box_2d` from the think-on stage and record
the flip as the finding — that alone answers your question, in the negative.

### Stage B — `think=on`, n=3

Per [ADR 0023](adr/0023-think-mode-is-per-model-and-measured-on-policy.md),
n=3 for think-on because on-policy sampling is not deterministic. Surviving
configurations × 7 geometries × 3. If both formats survive A: 4 configurations ×
7 × 3 = **84 cells**. If only named survives: 42.

The context ladder runs per SPEC H4a and **the rung reached is a first-class
result**, not a diagnostic — gemma4 under thinking is ~4× tokens, so expect
escalation and record where it settles.

### Stage C — production candidate, n=5

The single winning configuration, at `hd` + the two geometries where it was
weakest, n=5. This produces the numbers you would actually quote: JSON validity
rate, salvage rate, order-correctness, IoU, `num_ctx` rung, and latency.

---

## 3. What each cell records

Already captured: `prompt_eval_count`, `eval_count`, `gen_tps`, `prefill_tps`,
`total_duration`, `load_duration`, `prompt_eval_duration`, `eval_duration`,
`req_num_ctx`, `req_num_predict`, `sampling_source`, `powermode`, and — since the
geometry work — `geometry`, `image_size`, `image_aspect`, `label_px_clamped`.

Added by P1/P2: `thinking_chars`, `answer_chars`, `thinking_tokens_est`,
`answer_tokens_est`, `json_salvaged`, `salvage_method`.

Contract metrics per cell: `declared_type`, `declared_order`, `coord_order`
actually used, `anchor_implied_type`, `anchor_implied_ref`, `self_check`,
`hits_anchor`, `hits_declared`, `hits_bestfit`, `bbox_mean_iou`,
`degenerate_boxes`.

**The decisive one is order-correctness against ground truth**, not
`declared_order` — the whole point of C2 is that the declaration is exactly what
cannot be trusted.

---

## 4. Decision rule

Ship a configuration only if, across **all 7 geometries**:

1. axis order correct in every cell (not "declared correct" — *correct*)
2. JSON valid, or salvaged by steps 1–2 only; step 3–4 salvage is a warning sign
3. `self_check` passes, **and** `hits_anchor` = 6 where the anchor is on — a
   `self_check` pass with `hits_anchor` < 6 is the silent failure C7 warns about,
   and the geometry sweep found four of those in fourteen qwen3.6 cells
4. the `num_ctx` rung is stable across repeats, so a production window can be set

If `think=on` fails 1 or 2 while `think=off` passes, the honest recommendation is
think-off with the reason recorded — your stated preference for thinking is not
worth an untrustworthy box.

## 5. Cost and what is not covered

Cell counts: A=28, B=42–84, C≈15 — 85–127 total. Wall-clock is **not estimable yet** — there is
no throughput baseline for gemma4 on this build or platform, which is what P0
produces. Estimate after preflight, not before.

Not covered: one model, one quantisation, one platform, **one runtime**. Nothing
here transfers to `gemma4:31b-nvfp4` (different runtime *and* an unstable one) or
to the 26b family — and the 26b family is exactly where the
order flips were measured, so do not generalise this result to it. `box_2d`
against ground truth is measured only on the synthetic shape scene; a photograph
with ambiguous object boundaries is a different test.
