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

## Outcome (2026-10-04 and 2026-10-05)

- **Upstream merged #29941 and closed #27044.**
  - The upstream CUDA maintainer merged [#29941](https://github.com/ggml-org/llama.cpp/pull/29941) at 12:10 UTC
    as `dd266785c`.
  - They closed [#27044](https://github.com/ggml-org/llama.cpp/pull/27044) at 12:12, "in favor of the other one,
    assuming the issue is now fixed".
- **What merged is the one-line `ne11` → `ne12`** measured in "Results on sm_120" below.
- **It is not the fix.** Master `dd266785c` as shipped aborts on a `test-backend-ops` case below 128 tokens.
  [upstream-mmq-successor-material.md](upstream-mmq-successor-material.md) has the case, a CPU check of every
  rule, and a change that covers them.
  - #27044's own line was not the whole fix either: it was short with one expert per token, and padded no
    `ids_dst`.
- **The fork switched 903 to that change on 2026-10-05,** the maintainer's decision: pad for the widest tile,
  as before #24127.
- **Nothing was posted upstream by the fork.** A successor PR is the maintainer's to write.

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

## The review, and the maintainer's variant (2026-10-04)

Facts for the maintainer's reply. As above, the reply itself must be written by hand.

**State of #27044 on the morning of 2026-10-04:** open, with review required. It was closed that day; see
"Outcome" above.
- Other users confirmed the fix on sm_75 (RTX 2080 Ti), on sm_120 (twice) and on a GB10
  (sm_121a).
- An upstream reviewer linked issue #29847, which has a `test-backend-ops` reproducer:
  `test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 512, 10, b, 640, 508, 2560)`, for `b` false
  and true. On the GB10 our line takes `compute-sanitizer` memcheck from 485 errors to 0, and the
  full `MUL_MAT_ID` (933) and `MUL_MAT` (1304) suites pass. The reviewer suggested adding the two
  cases to our PR.
- **The upstream CUDA maintainer** wrote #24127, the refactor that introduced the bug. He said
  our fix "looks 90% correct" and opened his own PR,
  [#29941](https://github.com/ggml-org/llama.cpp/pull/29941). He asked whether #29941 also
  works for us. #29941 changes the same line, but pads with `ne12` (the tokens in the batch)
  instead of `ne12*n_expert_used`. It adds no test cases, because the tensors are large.

**How the two lines differ.** This comes from reading upstream `master` (`mmq.cu`, `mmq.cuh`) on
2026-10-04, not from a measurement:
- **The maintainer's argument holds.** The `ids` launch passes `ncols_max = ne12`, and
  `mul_mat_q_switch_J` picks the tile width `J` against `ncols_opt`. That is `ne12` on NVIDIA.
  On RDNA3 and RDNA4, gfx1151 included, it is the average number of tokens per expert,
  `⌈ne12·n_expert_used/ne02⌉`, which is never larger. So `ne12` bounds the launched `J`.
- **From 128 tokens up, both lines give the same padding on NVIDIA.** `ggml_cuda_mmq_get_J_max` caps
  its argument at 512, but no tile wider than 128 has a config, so both lines pad 128 blocks
  (corrected 2026-10-04 from "512 tokens": see "Results on sm_120"). Every crash reported on #27044
  used a batch of 1024 to 2178 tokens, and ours used 2040. So #29941 fixes all of them by arithmetic.
- **Below 128 tokens, #29941 pads less, and it can pad too little.**
  - `ggml_cuda_mmq_get_J_max` rounds its argument down to a multiple of 8.
  - `mul_mat_q_switch_J` picks the smallest `J` that covers `ncols_opt` in one tile, so it rounds
    up.
  - The src1 tile load, `for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; ...)`, has no bound.

  With 100 tokens on sm_120, #29941 pads 96 blocks. No config exists at `J = 104`, so the launch
  takes `J = 112`. The last tile can then read up to 15 blocks past the padding; that was measured
  ("Results on sm_120"). `ne12*n_expert_used` covers this whenever a token uses two or more experts.
- **Neither line closes the two gaps** that the sanitizer report on #27044 (2026-09-02) found:
  - fewer than 8 rows gets no padding, while a `J = 8` kernel loads 8;
  - `ids_dst` is read past its end by up to `(J-1)*4` bytes.

  Padding by the `J` that is actually launched closes all three gaps.

  On Turing and newer, MMQ takes only `MUL_MAT_ID` batches above MMVQ's limit: 8 tokens for most
  types, 7 for q2_K, 5 for q3_K (`get_mmvq_mmid_max_batch_turing_plus`). So with two or more experts
  per token, the MMQ path always has at least 12 rows, and the first gap needs one expert per token.

**What this means for the fork.** Compat `903` carries our line, and every production build
applies it, gfx1151's included. gfx1151 reaches this branch:
[rocm-mmq-ids-padding-result.md](tasks/rocm-mmq-ids-padding-result.md). If upstream merges #29941
instead of #27044, 903 stops applying at the next llama.cpp bump. Then decide between two
options:
- drop 903, because #29941 covers every shape that we have seen crash;
- keep a reworked 903 on top of #29941, for the rounding gap.

**Open, for the CUDA host:**
1. ~~On sm_120, run each case three ways: `master`, `master` + 903, and `master` + #29941, under
   `compute-sanitizer --tool memcheck`.~~ Done on 2026-10-04: see "Results on sm_120" below.
2. ~~Give the maintainer the results, for a hand-written reply on #27044.~~ Given on 2026-10-04. #27044 was
   closed the same day; see "Outcome" above.
3. At the first llama.cpp pin that contains `dd266785c`, re-cut 903's ids hunk against its `ne12` line. The
   change stays the widest-tile padding.

## Results on sm_120 (2026-10-04)

**Setup.**
- **Hardware and llama.cpp:** the CUDA host's RTX PRO 6000 Blackwell (sm_120), CUDA 13.0, driver 580.126.18,
  llama.cpp master `05043961`.
- **Three builds of `test-backend-ops`,** which differ only in the ids-path padding argument:
  - `master`: `ne11`;
  - #29941: `ne12`;
  - #27044, that is 903's line: `ne12*n_expert_used`.
- **One process per case,** under `compute-sanitizer --tool memcheck`.

**#29941 reads past its buffer below 128 tokens. #27044 does not, in any case.** Memcheck errors when the src1
buffer has its own exact-size allocation (✓ test passed, ✗ aborted):

| case | `master` | #29941 | #27044 |
|---|---|---|---|
| 65 tokens, 576 rows (fallback), q4_K, 256 experts, 8 used | 3,041 ✗ | **3,229 ✗** | **0 ✓** |
| 100 tokens, 512 rows, q4_K, 256 experts, 8 used | 6,693 ✗ | **557 ✗** | **0 ✓** |
| 100 tokens, #29847's shape (q4_0, 512 experts, 10 used) | 7,341 ✗ | **53 ✗** | **0 ✓** |
| 2040 tokens, the original fault's shape | 101 ✗ | 0 ✓ | 0 ✓ |
| 508 tokens, #29847, `b=0` | 913 ✗ | 0 ✓ | 0 ✓ |
| 508 tokens, #29847, `b=1` | 2,397 ✗ | 0 ✓ | 0 ✓ |

With the stock memory pool, every cell is 0 ✓ except `master` on #29847's two cases: 237 ✗ for `b=0` and 1,257 ✗
for `b=1`.

**What decides each result.**
- **The tile the kernel launches, against the padding.** `mul_mat_q_switch_J` sets the tile width `J`, and
  `ggml_cuda_mmq_get_J_max()` sets the padding.
  - At 65 tokens with 576 rows, the fallback configs offer only `J` = 8, 16, 32, 64 and 128, so the launch is
    `J = 128`. #29941 pads 64 blocks there, and #27044 128.
  - At 100 tokens with 512 rows, the launch is `J = 112`, against padding of 96 and 128 blocks.
  - The last expert's last tile can read up to `J − 1` blocks past the data.
- **The faults match the arithmetic.** #29941's 65-token errors come from `mul_mat_q<q4_K, 128, fallback>` reading
  past an allocation of 1,207,296 bytes. That is 1,198,080 bytes of data plus #29941's 64 blocks of 144 bytes.
- **At 508 and 2,040 tokens, both lines pad 128 blocks,** the widest tile the kernel launches. So #29847's
  cases and the original fault do not separate them.

**How to read the numbers.**
- **An error count does not measure the overrun.**
  - Memcheck stops each warp at its first out-of-bounds load. Every variant's furthest reported read is about
    1 KB past its allocation, also with `--padding 65536`.
  - So a count reflects the random routing and the scheduling. Two runs of the same #29941 build gave 2,337 and
    3,361.
  - Only zero against non-zero carries information.
- **The stock pool hides the over-read.** The next pool allocation is mapped right after the buffer, so the stray
  reads land in valid memory. That is why the exact allocation is needed.

**Two local changes, common to all three builds:**
- **The debug switch.** `MMQ445_EXACT=1` gives the src1 buffer an exact-size `cudaMalloc`, and is debug only.
- **A workaround for a launch failure in these builds.** In these builds of master `05043961`, every MoE `MUL_MAT_ID`
  through MMQ failed on this host with `invalid argument` at the `mm_ids_helper` launch, before any padding code
  ran. Master's own test cases failed the same way.
  - The compiled helper reports 1 KB of static shared memory (`cuobjdump -res-usage`). `CUDA_SET_SHARED_MEMORY_LIMIT`
    raises its dynamic limit to the device maximum, 101,376 bytes.
  - A standalone kernel with a 1 KB static array gets the same error.
  - The workaround leaves the limit at its default, which covers the `n_tokens*4` bytes of every case here.
  - **This is not shown to be upstream's problem** (corrected 2026-10-04):
    - `mmid.cu` is byte-identical in b11081, b11232 and master `05043961`.
    - Production's b11081 build runs this path on the same card.
    - Production's `sm_120a` PTX for the helper declares no static shared memory; it is compiled to SASS on the GPU
      at load time.
    - These builds compiled native `sm_120a` code with CUDA 13.0, as a static build. The 1 KB more likely comes from
      that build configuration than from upstream's code.

**A GPU-free check.** [`tasks/mmq-ids-padding-test.cu`](tasks/mmq-ids-padding-test.cu) replays the padding rule
against upstream's own config functions.
- **What it covers:** 2,439,360 shapes, from 4 architectures, 10 types, both fallback modes, every batch that takes
  MMQ up to 1,024 tokens, 2 to 16 experts used, broadcast or not.
- **What it finds:** #29941 leaves 268,440 of them short, by up to 63 blocks, all below 128 tokens. #27044 leaves none.
- **A correction:** its first version assumed MMVQ takes every batch up to 8 tokens. It does not for q2_K (7) or
  q3_K (5), so it missed 960 shapes. The corrected run, with the same conclusion, is the one above.
- **Its limit:** it replicates `mul_mat_q_switch_J`'s selection loop, so it holds only as long as upstream keeps that
  loop.

**Reproducing.**
- **CPU, no GPU:** build [`tasks/mmq-ids-padding-test.cu`](tasks/mmq-ids-padding-test.cu) with nvcc, from the root of a
  llama.cpp checkout. Its header has the command.
- **GPU:** run `tasks/mmq-ids-padding-gpu.sh <llama.cpp checkout>`
  ([script](tasks/mmq-ids-padding-gpu.sh)).
  - It applies [`tasks/mmq-ids-padding-gpu.patch`](tasks/mmq-ids-padding-gpu.patch), which holds the six cases, the
    debug switch and the workaround.
  - It builds the three variants and runs the matrix.
  - Any NVIDIA GPU that takes the MMQ path will do. Each case needs about 1 to 2 GB.

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
| Confirmed on | `f8def7fe1` (b10353) |
| Clean on | `cb295bf59` (b9888), 3/3 cold trials, same host and config — a regression, bisectable in b9888..b10353 |
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
- **There IS a GPU-free regression test**, which is worth offering if a maintainer asks for
  one. `get_J_max` and `ggml_cuda_mmq_get_config` are pure `__host__` functions of
  `(type, J, fallback, cc)` — no `ggml_cuda_info()`, no device queries — so the degenerate
  range is assertable on a CPU-only runner. `tasks/jmax-padding-gate.cu` compiles against the
  real `mmq.cuh` and, run with `CUDA_VISIBLE_DEVICES=""`, reports **1380 of 2240** cases
  returning zero padding across 14 architectures (Pascal→Blackwell, RDNA2–4, CDNA2/3, Vega),
  5 quant types, both `fallback` values, `ne11` in 1..16.

  Two things to keep straight when offering it. The zero range reaches `ne11 = 15`, not the
  1..7 that `ret -= ret % 8` alone predicts — for some `(type, cc, fallback)` no valid config
  exists at `J = 8` either, so the loop steps down to zero. And "`get_J_max` must return > 0"
  is a *contract* claim upstream may reject, since the function can legitimately mean "no tile
  fits"; the defensible invariant is at the call site — the `ids` branch must size its padding
  from the row count it actually writes. Offer the host test as a regression net, not as proof
  that `get_J_max` is itself wrong.

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
