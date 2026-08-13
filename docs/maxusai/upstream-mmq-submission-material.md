# Upstream submission material — facts to write from

Companion to [[upstream-mmq-ids-padding-issue]]. That document is the full investigation.
This one holds the facts, measurements and file references needed to submit the fix upstream.

> [!IMPORTANT]
> **This is raw material, not text to post.** ggml-org/llama.cpp prohibits AI-written posts,
> and an earlier version of this file offered paste-ready prose. That was wrong, and it was
> caught in production — see "What happened when we filed it" below.
>
> From `CONTRIBUTING.md`:
>
> > It is strictly prohibited to use AI to write your posts for you (bug reports, feature
> > requests, pull request descriptions, Github discussions, responding to humans, ...).
> >
> > Undisclosed AI usage may result in your account being permanently banned from
> > contributing to the project.
>
> AI-generated **code** is allowed with disclosure. AI-generated **prose** is not. Write the
> PR description, commit message and any replies yourself, from the facts below.
>
> `AGENTS.md` adds two requirements worth reading before submitting: you must be able to
> explain every line to a reviewer without AI assistance, and verbose AI-sounding responses
> "will not be well-received".

## What happened when we filed it

Recorded because the failure mode is not obvious and the next person will hit it.

PR [ggml-org/llama.cpp#27044](https://github.com/ggml-org/llama.cpp/pull/27044) was opened
2026-08-13 with an AI-written description and commit message. `ggml-gh-bot` flagged it within
hours on two counts: the PR template was not filled in, and the description and commit
message were AI-generated. Both had to be rewritten by hand, the commit amended and
force-pushed, and an AI-usage disclosure added.

The one-line code change was never the problem. The prose was.

Two further notes from that round:

- **The template is mandatory** and has a `## Requirements` section that must not be deleted,
  including an explicit `AI usage disclosure:` line. Fill it in honestly — disclosure is what
  keeps the account safe; concealment is what gets it banned.
- **The symptom text belongs first**, ahead of any analysis. Our first draft led with the
  code argument because that is what a reviewer needs, and omitted the error text entirely —
  so the report was unfindable by anyone searching the crash they were hitting. That is also
  how these reports find each other: #22867 was linked to this bug by its distinctive
  `find_slot` line, not by any argument. The filed PR had to be corrected after the fact.

## Strategy: a PR, not an issue

The fix is verifiable by reading one function — no hardware, no model, no reproduction.
Everything that makes an issue hard to action here (upstream cannot load our GGUF,
`test-backend-ops` goes green, the cold-vs-warm reproduction is fiddly) stops mattering when
the reviewer only has to check an internal inconsistency. Two prior reports of what may be
this same bug (#19705, #24399) were closed "not planned" after asking maintainers to
reproduce an MoE crash on hardware they may not have.

Order: PR first, then optionally a short comment on each of the two genuinely
MMQ/`mul_mat_id` threads pointing at it. #22867 and #22032 are weaker links; a comment there
would be noise.

---

## The facts

### The defect

- File: `ggml/src/ggml-cuda/mmq.cu`, `ggml_cuda_mul_mat_q()`, the `ids` branch (line ~206 on
  master at `a94d563`).
- The `src1_q8_1` allocation is a data term plus an MMQ tail-padding term.
- Data term is sized `ne12*n_expert_used`. Padding term is sized from `ne11`. Different
  quantities.
- The correct value is computed eight lines below as `ne11_flat = ne12*n_expert_used`, and is
  what the quantise kernel is given.
- `ggml_cuda_mmq_get_J_max()` (`mmq.cuh:360`) with `ne11 == 1`: `min(1,512) = 1`, then
  `1 - 1%8 = 0`, loop never runs, returns **0**. No tail padding at all.
- MMQ processes rows in tiles of up to `J_max = 512` and reads past the logical end.
- MoE gate/up broadcast activations across experts, so `ne11 == 1` — the `dedup_bcast` case
  the branch handles (`mmq.cu:193`).
- `ffn_down` is affected less: `ne11 = n_expert_used = 8`, padding sized for 8 rows.
- The `!ids` branch above passes `ne11` correctly, because there `ne11` *is* the row count.

### The fix

```diff
-        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
+        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12*n_expert_used) * sizeof(block_q8_1_mmq);
```

### The symptom, verbatim

```
decoding image batch 1/1, n_tokens_batch = 2040
find_slot: non-consecutive token position 4 after 3 for sequence 0 with 2040 new tokens
ggml-cuda.cu:106: CUDA error
ggml_cuda_compute_forward: MUL_MAT_ID failed
CUDA error: an illegal memory access was encountered
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2374
```

Server-side: HTTP 500,
`{"error":"an error was encountered while running the model: CUDA error\nCUDA error: an illegal memory access was encountered"}`

The `find_slot` line comes from `llama-memory-recurrent.cpp` and also appears on runs that do
**not** crash — it marks the path, not the fault.

### The observed fault

With `cudaStreamSynchronize(ctx.stream())` before the existing `cudaGetLastError()`, so the
error is attributed to the node that caused it rather than a later dispatch:

```
op=MUL_MAT_ID name=ffn_moe_gate-3
  dst  [512, 8, 2040, 1]
  src0 blk.3.ffn_gate_exps.weight : q4_K [2048, 512, 256, 1]
  src1 attn_post_norm-3 (reshaped): f32  [2048,   1, 2040, 1]
  src2 ffn_moe_topk-3             : i32  [   8, 2040,    1, 1]
branch=mmq ne0=512 ne1=8 ne2=2040 ne02=256 ne11=1 ne12=2040 type=q4_K
```

`ggml-cuda.cu:2371` is a bare `cudaGetLastError()` after dispatch — asynchronous and sticky —
so without that synchronise the reported op is where the error was *noticed*, not where it
occurred.

### Measurements

| | |
|---|---|
| Environment | RTX PRO 6000 Blackwell (sm_120), CUDA 13.0 |
| Also reproduced on | `cb295bf59`, `b4d6c7d8f`, `f8def7fe1` |
| Unaffected by | `GGML_CUDA_DISABLE_GRAPHS=1` (env verified reaching the runner) |
| Verification | 4 cold runs patched, no fault; 2 unpatched controls from the same tree, both fault |
| Added allocation | ≤ `512 * sizeof(block_q8_1_mmq)` = **72 KB per call**, constant, does not scale with `ne12` |
| `sizeof(block_q8_1_mmq)` | 144 bytes (`QK8_1_MMQ + 4*sizeof(half2)`, `mmq.cuh:56`) |

### Caveats a reviewer will want stated

- **`test-backend-ops` does not catch it.** The over-read lands in padding rows the kernel
  discards, so output is byte-identical and NMSE is unaffected. `compute-sanitizer --tool
  memcheck` does flag it. An independent sanitizer run is the most useful confirmation.
- **A passing run is not evidence of absence.** The overrun is a fixed size past the end and
  faults only when it crosses an unmapped page: the same request crashes on a fresh pool and
  passes once that pool has served a larger allocation.
- **Reproduction needs a non-ollama GGUF.** Upstream cannot load ollama's qwen3.6
  (`qwen35moe.rope.dimension_sections has wrong array length; expected 4, got 3`).

### Related issues — flag, do not claim

Not reproduced here, so these are hypotheses:

- **#24399** — sm_120 `mul_mat_q` out-of-range shared-memory store; same file, same kernel
  family, same hardware class. Attributed there to Blackwell int8-MMA codegen. Both
  workarounds offered are also explained by this defect: `GGML_CUDA_FORCE_CUBLAS=ON` skips
  MMQ entirely, and requantising changes `y_block_size`/`ne10_padded` and hence the
  allocation size.
- **#19705** — Qwen3-Coder-Next `ggml_mul_mat_id` assertion; MoE, `-ngl 99` only, fine on
  CPU, which is what a fault confined to this CUDA allocation looks like. Already treated
  there as MoE-wide (Nemotron-3-nano, gpt-oss-120b).
- **#18331** — Blackwell MUL_MAT illegal access, attributed to nvcc O3 codegen; its
  `-DCMAKE_CUDA_ARCHITECTURES=89` workaround changes both codegen and the fatbin, so it
  cannot separate a codegen fault from an allocation-layout one.

## Mechanics

Fork, branch from `master`, one-line change, push to the fork, PR against `ggml-org:master`.
`gh repo fork` leaves the clone's `origin` pointing at upstream — add the fork as a separate
remote or the first push 403s. Set `user.email` in the clone; the repo default may not be the
address you want on a public commit.
