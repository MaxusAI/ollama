# llama.cpp#29953 and the two gaps it leaves: facts to write from

Companion to [upstream-mmq-successor-material.md](upstream-mmq-successor-material.md). That file was written
against master `dd266785c` (#29941 merged) and argued for padding by the widest tile. This file supersedes its
arithmetic: **the required src1 padding is larger than every rule proposed so far assumed, including ours**, and
#29953 is the right shape of fix but still short in two places.

> [!IMPORTANT]
> **This is raw material, not text to post.** ggml-org/llama.cpp prohibits AI-written posts (bug reports, pull
> request descriptions, replies, ...), and undisclosed AI use can get the account banned. The maintainer writes
> the reply and any PR description by hand, from the facts below. The code here (the amendment, the test cases,
> the checks, the harness) is AI-generated, which the project allows **with disclosure**: fill in the template's
> `AI usage disclosure:` line.

## Upstream got there first: #29953 now carries both amendments (2026-10-05, 18:55Z)

**Everything below about src1 and `ids_dst` is now fixed upstream, found independently.** On #29953, `hclsys`
reported from a GB10 (sm_121a, CUDA 13.0) exactly the two gaps this file was written about, with the same
formula:

- src1: "the `tile_y` load loops in `mul_mat_q_process_tile` run to the next multiple of `nthreads` (J=112: 4032
  ints -> 4096 with 256 threads) and `mmq_get_nbytes_shared` pads `nbs_y` for that, but the global padding is
  only `J_best*sizeof(block_q8_1_mmq)`" -- 10 of 16 ids-shuffle seeds hit it at `n = 100`, 0 of 16 with
  `GGML_PAD(J_best*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int))`.
- `ids_dst`: "gets read up to `J_best-1` ints past `ne_get_rows` too with exact-size allocs".

The author pushed both the same day. At head `3070d927f` ("CUDA: fix MMQ out-of-bounds reads"):
`nthreads_best` is carried out of the selection loop, `src1_q8_1_padding` is the padded tile, and `ids_dst` is
`ne_get_rows + J_best-1`. That is our `tasks/mmq-amend-29953.patch` in all but spelling -- theirs uses the tight
`J_best-1` bound for `ids_dst` where ours used `J_best`. **So there is nothing to send upstream about those two,
and `mmq-amend-29953.patch` is of historical interest only.** Our independent arrival at the same formula, from a
different direction (the CPU sweep rather than a seed sweep), is a cross-check on both.

### RETRACTED: the NVFP4 y scales are not read out of bounds

This file previously claimed that `src1_scale` is read up to `J_best - 1` floats past its end, by the same
mechanism as `ids_dst`, and offered a two-line patch for it. **That claim does not survive measurement. Do not
pass it on.**

The reasoning was: `offset_y_scale += col_low + jt*J; y_scale_tile = y_scale + offset_y_scale;`, and the stream-k
fixup write-back is called as `write_back(sum, ids_dst, tmp_fixup + blockIdx.x*(J*I), y_scale, I, I, J)` with
`j_max == J`, so its `if (j > j_max)` guard never fires and `y_scale_tile[j]` is read for the whole tile. Every
part of that is in the source at head `3070d927f`, and the write-back is the same
`ggml_cuda_mmq_write_back_mma<type, J, fallback>` for NVFP4 as for everything else, so the
`if constexpr (type == GGML_TYPE_NVFP4)` branch is compiled.

What was measured instead, on the RTX PRO 6000 Blackwell with a `120a` build -- native FP4 really is in play, the
build reports `prec=q4` and `MMQ_SCALE alloc n=9` shows the buffer being allocated:

| shape | runs the fixup kernel? | `src1_scale`, exact size, memcheck | `guard:src1_scale` |
|---|---|---|---|
| `nvfp4, 8 experts, 1 used, 640x9x512` | **yes** (40 tiles, 21% of the waves on 188 SMs) | **0 errors** | 3/3 pass |
| `nvfp4, 256 experts, 16 used, 640x16x512` | no (2560 tiles, 97%, so blocks == ntiles) | 0 errors | 3/3 pass |

The control that rules out a harness fault: shrinking the allocation to one float makes memcheck report 9 errors
immediately -- as **writes**, from the quantize kernel filling one scale per column. So the buffer is
instrumented, it is live, the kernel's results are numerically correct, and nothing reads or writes past its
`ne12*n_expert_used` entries.

**Two things this corrects in the earlier write-up.** First, "every NVFP4 config uses stream-k, so the path is
not exotic" was beside the point: stream-k is always selected, but the *fixup kernel* -- the only unbounded
reader -- runs only when `launch_mul_mat_q` falls back to `nsm` blocks, which it does only when the tiles fill
under 90% of the waves. The first shape tried was at 97% and never ran it;
[tasks/mmq-fixup-shape-search.cu](tasks/mmq-fixup-shape-search.cu) finds shapes that do. Second, the claim was
labelled "read, not measured" on the grounds that this host has no native FP4. That was wrong too:
`blackwell_mma_available()` only needs `highest_compiled_arch >= 1200`, which even a plain `sm_120` build gives,
and `__CUDA_ARCH_LIST__` is 1200 for both `120` and `120a`.

**Why the read does not happen is not established.** The source says the whole tile is read; the device says
nothing goes past the buffer. Rather than guess at the reconciliation, the claim is withdrawn.
`tasks/mmq-amend-29953-yscale.patch` is kept only as the record of what was tried and is **not** to be sent
anywhere. There is therefore **nothing outstanding for upstream from this work**: #29953's head fixes src1 and
`ids_dst`, and the third buffer turned out not to need it.

## Where it stands (2026-10-05)

- **#29953** ("CUDA: fix MMQ ncols rounding direction for alloc", head `5bd8b0013`, base `46847e615`) moves the
  tile-size choice ahead of the allocation and pads by exactly the `J_best` it is about to launch, so the two can
  no longer round in opposite directions. It also deduplicates `fallback` into
  `ggml_cuda_mmq_needs_fallback()` and caps `J` at 128 in the config `static_assert`.
- Its author says they could not reproduce further issues locally and asked for a reproduction.
- **Reproduced, twice over.** #29953 still faults, on two separate reads:
  1. **src1 is padded by one tile's worth of blocks, but a tile's y load reads a *padded* tile.** The loop is
     `for (l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += nthreads) tile_y[l0 + tid] = by0[l0 + tid];` with no bound on `l`,
     so it reads `GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int))` bytes from the tile's first column --
     the same expression `mmq_get_nbytes_shared()` already uses to size the shared-memory tile it copies into.
     That is more than `J` blocks whenever `nthreads*sizeof(int)` does not divide `J*sizeof(block_q8_1_mmq)`.
  2. **`ids_dst` is not padded at all**, by any rule so far. A tile loads `J` entries,
     `ids_dst_shared[j] = ids_dst[col_low + jt*J + j]` for `j < J`, again unbounded, so the last expert's last
     tile reads up to `J - 1` entries past its `ne12*n_expert_used`.
- Both are plain out-of-bounds reads of device memory; neither changes a result, because the write-back paths stop
  at `j > j_max`. They fault only when the read leaves mapped memory, which the ggml memory pool usually prevents
  -- see "Why nobody sees this locally".

## The arithmetic, corrected

For a launch of tile width `J` with `nthreads` threads, write `B = sizeof(block_q8_1_mmq)` (144) and
`T = GGML_PAD(J*B, nthreads*sizeof(int))`, the bytes one tile's y load copies from global memory.

src1 is laid out `[k-plane][column]`, so in the last k-plane a tile whose first column is `c0` reads to
`c0*B + T`. With `P` columns in the buffer and the padding counted in blocks:

```
pad_blocks  >=  ceil(T / B) - 1          (worst case c0 = P - 1: the last tile holds one column)
```

For MUL_MAT_ID that worst case is real at any batch size, because the last expert's last tile may hold a single
row. `J - 1` -- what every rule so far aimed at -- is only correct when `T == J*B`.

`ids_dst` is read `J` entries per tile from the expert's first row, so it needs `J - 1` entries.

On an RTX PRO 6000 Blackwell (sm_120) every q4_0 and q4_K config has `nthreads = 256`, so:

| `J` | `T` | src1 `need` | `J - 1` | #29953 pads | short by | where the row comes from |
|---|---|---|---|---|---|---|
| 8 | 2,048 | 14 | 7 | 8 | 6 blocks | the formula; reachable with q2_K at 8 tokens (its MMVQ limit is 7) |
| 16 | 3,072 | 21 | 15 | 16 | 5 blocks | printed by the build: `ids16` |
| 112 | 16,384 | 113 | 111 | 112 | 1 block | printed by the build: `ids64`, `j100_b0`, `t100` |
| 128 | 18,432 | 127 | 127 | 128 | covered | printed by the build: `e120`, `p29847`, `one113`, `fb65`, `orig2040` |

"printed by the build" means `MMQ_DEBUG_PRINT=1` ([tasks/mmq-debug-alloc.cuh](tasks/mmq-debug-alloc.cuh)) had the
build report its own `J`, `nthreads` and padding for that shape on the device; the lines are in
[tasks/mmq-successor-results/case-shapes.txt](tasks/mmq-successor-results/case-shapes.txt). 6 blocks is the worst
shortfall the CPU check finds for #29953 anywhere, across all five architectures.

`J = 128` is exactly covered, which is why the large batches that first exposed this bug are fine under both
#29941 and #29953, and why a reproduction has to use a batch that lands on a narrower tile.

## The rules side by side

The padding each rule gives src1, as the argument of `ggml_cuda_mmq_get_J_max()` unless stated:

| rule | src1 padding | pads `ids_dst`? | pads the NVFP4 y scales? |
|---|---|---|---|
| #24127 (b9992 .. `dd266785c^`) | `get_J_max(ne11)` -- 0 for broadcast gate/up | no | no |
| #29941 (master, `dd266785c`) | `get_J_max(ne12)` | no | no |
| #29953 | the launched `J_best`, in blocks | no | no |
| #27044 (our compat 903 until now) | `get_J_max(ne12*n_expert_used)` | no | no |
| #448 as published | `get_J_max(type, fallback, cc, 512)` -- the widest tile | yes | yes |
| **#448 amended** (recommended) | the widest **padded** tile, `max_J ceil(T/B)` | yes | yes |

## The CPU check: every rule, 4.7 million shapes

[tasks/mmq-rules-check.cu](tasks/mmq-rules-check.cu) replays the tile selection against llama.cpp's own config
functions, with no GPU. It covers sm_75, sm_86, sm_89, sm_120 and gfx1151; ten quantization types; fallback 0 and
1; every MMQ batch up to 1024 tokens starting above each type's MMVQ limit; `n_expert_used` 1 to 16 and broadcast
0 and 1; and on gfx1151 also 8, 32, 128 and 256 experts, since RDNA3 picks `J` against the tokens per expert.

> [!WARNING]
> **The gencode list is load-bearing.** `ggml_cuda_highest_compiled_arch()` reads `__CUDA_ARCH_LIST__`, and
> `ggml_cuda_mmq_get_config()` returns a different config table for an architecture the binary was not compiled
> for. Built without `-gencode`, the check reports `get_J_max(512) = 64` on sm_120 where the device uses 128, and
> every count comes out wrong. Build it with [tasks/build-check.sh](tasks/build-check.sh), which passes
> `sm_75`, `sm_86`, `sm_89` and `sm_120`. The first version of this check was built without them; the numbers in
> the companion file's "CPU check" section are from that build and are superseded by the ones below.

Shapes left short, from [tasks/mmq-successor-results/check-rules.txt](tasks/mmq-successor-results/check-rules.txt)
(4,717,824 ids-branch shapes, 101,600 dense):

| rule | src1, 1 expert/token | src1, >=2 experts/token | `ids_dst` | dense |
|---|---|---|---|---|
| #24127 (`ne11`) | 162,672 (worst 127) | 4,554,640 (worst 127) | all 4,717,328 (worst 127) | 705 (worst 9) |
| #29941 (`ne12`) | 11,410 (worst 63) | 365,300 (worst 63) | all 4,717,328 | - |
| **#29953** | **76,640 (worst 6)** | **1,531,586 (worst 6)** | **all 4,717,328** | **90 (worst 5)** |
| #27044 | 11,410 (worst 63) | 29,944 (worst 17) | all 4,717,328 | - |
| #448 as published | 128 (worst 5), gfx1151 q2_K only | 25,256 (worst 5), gfx1151 q2_K only | covered | 45 (worst 5), gfx1151 only |
| **#448 amended** | covered | covered | covered | covered |

> [!NOTE]
> **Two corrections from the ROCm host (MaxusAI/ollama#449).** The counts above are the corrected ones.
> 1. The NVIDIA MMVQ limit is not "Turing and newer": Volta, and Ada Lovelace and newer, always take MMVQ for
>    MUL_MAT_ID up to `MMVQ_MAX_BATCH_SIZE` for **every** type, and only Turing and Ampere use the per-type
>    table. Reading it as Turing-and-newer counted q2_K at 8 tokens and q3_K at 6..8 as MMQ shapes on sm_89 and
>    sm_120, where MMVQ takes them: 496 shapes, now excluded. #29953's worst case, `q2_K, J = 8`, therefore holds
>    on sm_75 and sm_86 but **not** on sm_89 or sm_120.
> 2. The gfx1151 shortfall is **q2_K only**, not q2_K and q3_K: q3_K's non-fallback table on gfx1151 reaches
>    `J = 128`, so its widest tile already covers the 85 blocks needed. `mmq-jpad-cost.cu` had been printing one
>    row all along and the earlier text said two types anyway.

"worst" is blocks (or `int32` entries for `ids_dst`) past the allocation. Per-architecture counts are in the
program's output; on sm_120 #29953 is short in 4,488 one-expert and 134,640 multi-expert shapes. gfx1151's column
comes from the host-side config tables as nvcc compiles them, not from a measurement on AMD hardware.

A count here is **shapes where the rule can be short**, not shapes that fault. The check asks the question a
padding rule has to answer -- is the allocation large enough for the worst routing this shape admits, one row in
the last non-empty expert's last tile -- and `ne12*n_expert_used` rows over `ne02` experts always admits it.
Whether a particular run reads past depends on its routing, on where the pool put the buffer, and on what the
pool mapped after it.

Note the direction of the trade: #29953 is short in **more** shapes than #29941 but by **less** -- at most 6
blocks instead of 63 -- because it fixes the rounding and leaves only the padded-tile term.

## Why nobody sees this locally

Both reads usually land in memory the process has already mapped, so they neither fault nor show up under
`compute-sanitizer`, which only knows the pool's own suballocation:

- `ggml_cuda_pool_vmm` bumps allocations from its base and maps in multiples of the allocation granularity,
  measured at 2 MiB on this card. A read a few hundred bytes past a buffer stays inside that mapping.
- In the ids branch the pool hands out `ids_src1`, `ids_dst`, `expert_bounds`, then `src1_q8_1`, so `ids_dst`'s
  over-read lands in `expert_bounds` and src1's lands in whatever follows -- often the stream-k fixup buffer,
  `nsm*J*I` floats (12 MiB on 188 SMs).
- This is why #29847's shape at 100 tokens shows no sanitizer errors on a stock build, and why the fault that
  started all of this looked like an intermittent crash gated on `num_ctx`.

Two debug switches make the reads observable. Neither is for upstream;
[tasks/mmq-debug-alloc.cuh](tasks/mmq-debug-alloc.cuh) holds both, selected by `MMQ_DEBUG_ALLOC`:

- `exact` gives `src1_q8_1` and `ids_dst` their own exact-size `cudaMalloc`, so memcheck reports the read. Needs
  the sanitizer.
- `guard` places each buffer so its **last byte is the last byte of its own virtual-memory mapping**, with the
  next granule reserved and never mapped (`cuMemAddressReserve` one granule long, `cuMemCreate`/`cuMemMap` only
  the rest). A read past the end then faults on the device. **No sanitizer, and deterministic.**

Either takes a suffix naming one buffer -- `guard:src1`, `guard:ids_dst` -- so a fault is attributable. The
guard's slack is the pointer alignment: 4 bytes for `ids_dst` (the kernel loads it with scalar `int` loads) and
128 bytes for src1, which is less than the 144-byte block an over-read takes.

## The reproduction

One `test_mul_mat_id` case separates all of it. It is small, it is in
[tasks/mmq-tests.patch](tasks/mmq-tests.patch) as `ids16`, and on sm_120 it has `J = 16`, `nthreads = 256`,
src1 `need` 21 blocks and `ids_dst` `need` 15 entries:

```
test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, /*n_mats =*/ 256, /*n_used =*/ 16, /*b =*/ false,
                /*m =*/ 640, /*n =*/ 16, /*k =*/ 2560)
```

`n_used = 16` with `b = false` makes `ne11 = 16`, so `get_J_max(ne11) = get_J_max(ne12) = 16`: every rule gives
src1 at least `J` blocks, and only the padded-tile term is missing. 256 experts over 16 tokens x 16 used leaves
the last non-empty expert holding well under one tile, so its last tile reads past `ids_dst`.

Run it three ways. The results below are from this host; the full matrix is under "Results".

| `MMQ_DEBUG_ALLOC` | what it exposes | `ne11` | `ne12` | #29953 | #27044 | #448 |
|---|---|---|---|---|---|---|
| unset (stock pool) | nothing -- the pool hides both | pass | pass | pass | pass | pass |
| `guard:src1` | the padded-tile term | abort | abort | **abort** | pass | pass |
| `guard:ids_dst` | the unpadded `ids_dst` | abort | abort | **abort** | abort | pass |

`#27044` passes `guard:src1` here only because `get_J_max(ne12*n_expert_used) = get_J_max(256) = 128`, far more
than the 21 blocks this shape needs; it is short elsewhere (see the CPU check).

**Why this shape.** `need` is the worst case over all routings. What a given case actually reads past the data
is `ceil(T/B) - r` blocks, where `r` is the last non-empty expert's row count, so the worst case needs `r = 1`.
Whether a case can realise it is set by the mean rows per expert, `ne12*n_expert_used / ne02`:

| case | rows | experts | mean rows/expert | `J` | realises the worst case? |
|---|---|---|---|---|---|
| `ids16` | 256 | 256 | 1.0 | 16 | yes, every run |
| `j100_b0`, `j100_b1` | 1,000 | 512 | 2.0 | 112 | sometimes |
| `t100` | 800 | 256 | 3.1 | 112 | sometimes |
| `ids64` | 6,400 | 256 | 25.0 | 112 | no: reads about 89 blocks past, which `ne12`'s 96 covers |

So `ids16` is the case to hand over: one row per expert, every run, and a 5-block shortfall against #29953 rather
than one. `ids64` is the control that shows the routing matters -- it passes `guard:src1` under every rule but
`ne11`. An over-read is a function of the routing as well as the shape, which is the third reason this bug is
hard to hit on purpose, after the pool and the fixup buffer.

## The amendment

[tasks/mmq-amend-29953.patch](tasks/mmq-amend-29953.patch) applies on top of #29953 (verified against
`5bd8b0013`), 8 changed lines, `mmq.cu` only:

- Carry the chosen config's thread count out of the selection loop as
  `nbytes_pad_y = GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int))`, and pad src1 by that
  instead of `J_best * sizeof(block_q8_1_mmq)`, in both branches. It is the expression
  `mmq_get_nbytes_shared()` already uses for the tile the load copies into.
- `ids_dst` gets `J_best` more entries.
- For NVFP4 with native FP4, `src1_scale` gets `J_best` more entries in both branches: the stream-k fixup
  write-back is called as `write_back(..., y_scale, I, I, J)`, so it reads `y_scale[j]` for all `J` columns of
  its tile. Found by reading the code; the GPU runs here use the q8_1 path, since an sm_120 build without `120a`
  has no native FP4.
- **Cost:** on sm_120 `nbytes_pad_y` is at most 18,432 bytes of src1 padding, 512 bytes of `ids_dst` and, for
  native FP4, 512 bytes of scales per call -- the same order as #29953's own padding, and less than what both
  branches reserved before #24127.
  - Against the plain widest-tile rule the amendment is free almost everywhere: across the five architectures and
    ten types checked, the two differ in **exactly one** combination -- gfx1151, non-fallback, q2_K, where the
    padding goes from 80 to **85** blocks, **720** bytes more per call. Everywhere else both give 128 blocks.
    [tasks/mmq-jpad-cost.cu](tasks/mmq-jpad-cost.cu) prints it.
    - The patch computes `J_pad = nbytes_pad_y / sizeof(block_q8_1_mmq)`, a **floor**, so it reserves 85 blocks
      and not the 86 a ceiling would give. That is sufficient -- `floor(T/B) >= ceil(T/B) - 1`, the requirement --
      and the CPU check now scores the floor, so the rule it validates is the one the code implements. The ROCm
      host measured `pad=85` on the device and the read at 12,288 - 144 = 12,144 bytes, inside it.

An equivalent standalone change against master, for a tree without #29953, is
[tasks/mmq-fix-amended.patch](tasks/mmq-fix-amended.patch): it takes the maximum padded tile over every config
that exists, so it does not depend on the launch choice.

**Worth offering alongside it, in #29953's own spirit.** The amendment spells `GGML_PAD(J*B, nthreads*4)` out in
`mmq.cu`, which leaves the same expression written twice -- once for the shared-memory tile, once for the
allocation -- and that duplication is how this class of bug arrives. Naming it once in `mmq.cuh` removes that:

```c
// Bytes of src1 a tile of this config reads from its first column: the tile load copies the whole padded
// shared-memory y tile, so this is exactly the y term of mmq_get_nbytes_shared().
static size_t mmq_get_nbytes_y_tile(const ggml_cuda_mmq_config & config) {
    return GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));
}
```

with `mmq_get_nbytes_shared()` calling it for its `nbs_y` term and `ggml_cuda_mul_mat_q` calling it for the
padding. That is a refactor of the amendment, not a different fix: `mmq_get_nbytes_shared()` already computes
`nbs_ids + nbs_x + GGML_PAD(nbs_y, config.nthreads*sizeof(int))` with `nbs_y = config.J*sizeof(block_q8_1_mmq)`,
so the helper returns the identical value. **It is not what the measurements here were taken with** -- they used
the 8-line form, which keeps `mmq.cuh` untouched -- so it is a suggestion for the maintainer to weigh, not a
tested change.

## A second architecture, and the dense branch (sm_75)

`dense321` is a plain `MUL_MAT` -- `test_mul_mat(Q2_K, F32, 512, 321, 1024)`, no `MUL_MAT_ID` anywhere. On sm_75
(RTX 2080 Ti, the only NVIDIA architecture where the CPU check finds the dense branch short at all, 45 shapes),
three rules x two cases x three modes, from [tasks/mmq-successor-results/matrix-sm75.md](tasks/mmq-successor-results/matrix-sm75.md):

| guard page | `ne12` (#29941) | #29953 as published | #29953 + amendment |
|---|---|---|---|
| `ids16`, src1 | 0/3 abort | 0/3 abort | 3/3 |
| `ids16`, `ids_dst` | 0/3 abort | 0/3 abort | 3/3 |
| **`dense321`, src1** | 3/3 | **0/3 abort** | 3/3 |
| `dense321`, `ids_dst` | 3/3 | 3/3 | 3/3 (the dense branch has none) |
| stock, both cases | pass | pass | pass |

- **The defect is not MUL_MAT_ID-specific.** A plain dense matmul aborts, deterministically, on a second
  architecture.
- **`ne12` passes `dense321` where the published #29953 fails**, because master's `get_J_max(ne11)` with
  `ne11 = 321` happens to give 128 blocks, covering the 85 needed, while #29953 tightened it to exactly
  `J_best = 80`. In the dense branch the published fixup was a regression against master for this shape.
- **Read against the published `5bd8b0013`, not today's head.** The head's padded-tile fix gives
  `GGML_PAD(80*144, 1024) = 86` blocks, so it covers this; the "amendment" column is equivalent for src1 and
  passes. So this is evidence that the correction upstream landed was necessary, and necessary beyond MoE -- not
  a live bug in the head.
- `ids16` reproducing identically on sm_75 is the check that the sm_120 results were not architecture-specific.

## Results

Seven padding rules x twelve cases x four modes on one RTX PRO 6000 Blackwell (sm_120, CUDA 12.8 throughout),
one process per run, from [tasks/mmq-rules-gpu.sh](tasks/mmq-rules-gpu.sh). Full table rendered by
[tasks/mmq-rules-table.py](tasks/mmq-rules-table.py): [tasks/mmq-successor-results/matrix.md](tasks/mmq-successor-results/matrix.md),
raw rows in [results.tsv](tasks/mmq-successor-results/results.tsv).

**`ids_dst`, guarded, all twelve cases.** `ne11`, `ne12`, #29953 and #27044 abort in every case; the three rules
that pad `ids_dst` pass every case. This is not a corner: it is every MoE shape tested.

**Exact-size allocations under memcheck, all twelve cases.** All four published rules report errors, from 43
(`orig2040`) to 11,939 (`one113`, #29953); the three amended rules report 0 in all twelve.

**src1, guarded.** Here the rules separate, and the routing matters:

| case | `ne11` | `ne12` | #29953 | #27044 | the three amended |
|---|---|---|---|---|---|
| `ids16` | abort | abort | **abort 2/2** | pass | pass |
| `j100_b1` | abort | abort | **abort 1/2** | pass | pass |
| `one113` | abort | abort | pass | **abort 2/2** | pass |
| `j100_b0`, `e120_b0/b1`, `fb65`, `t100` | abort | abort | pass | pass | pass |
| `p29847_b0/b1`, `orig2040`, `ids64` | abort | pass | pass | pass | pass |

**The `J = 112` cells, ten repeats each, with the amendment as a control in the same harness.** #29953's
shortfall there is one 144-byte block, which only bites when the routing puts a single row in the last non-empty
expert, so it is a rate rather than a verdict. Aborts out of ten:

| case | mean rows/expert | `ne12` (#29941) | **#29953** | **#29953 + amendment** |
|---|---|---|---|---|
| `ids16` (`J` = 16, 5 blocks short) | 1.0 | 10/10 | **10/10** | **0/10** |
| `j100_b1` (`J` = 112, 1 block short) | 2.0 | 10/10 | **4/10** | **0/10** |
| `j100_b0` (`J` = 112, 1 block short) | 2.0 | 10/10 | **1/10** | **0/10** |

So the one-block shortfall is real and reproducible, at 4 in 10 and 1 in 10; the amendment is clean 10 of 10 on
both in the same harness, which is what separates the padding from the harness. `ids16` is the case to hand over:
its 5-block shortfall and one row per expert make it deterministic.

**Combined guard, both buffers at once, five cases x seven rules.** All four published rules abort on all five
cases; all three amended rules pass. No exceptions.

**Stock, no sanitizer, no debug allocator.** `ne11` aborts on four cases and `ne12` on `e120_b0` and `e120_b1`;
everything else passes under every rule. That is the pool doing what it always does, and the reason a user hits
this as an intermittent crash rather than a test failure.

749 rows in all, from the main pass plus [tasks/mmq-rules-addenda.sh](tasks/mmq-rules-addenda.sh).

## Reproduce

- **CPU only, nvcc and no GPU**, from a llama.cpp checkout at `dd266785c` or later:
  `tasks/build-check.sh <checkout> tasks/mmq-rules-check.cu ./mmq-rules-check && ./mmq-rules-check`.
  The exit code is non-zero only if the amended rule leaves a shape short. `--design` lists shapes where only
  `ids_dst` is short.
- **GPU:** `tasks/mmq-rules-gpu.sh <llama.cpp checkout at dd266785c> [out dir]`. It applies the test cases,
  builds each rule twice (with and without the debug allocator) via
  [tasks/mmq-variant.py](tasks/mmq-variant.py), and runs every case stock, under `guard:ids_dst`, under
  `guard:src1`, and under memcheck with exact-size allocations. It is resumable.
- **The shortest path for a reviewer**, on any NVIDIA GPU:
  1. apply `tasks/mmq-tests.patch` to #29953's branch,
  2. apply `tasks/mmq-variant.py`'s debug allocator (`mmq-variant.py <checkout> p29953 --debug`),
  3. `MMQ_DEBUG_ALLOC=guard:src1 test-backend-ops test -o MUL_MAT_ID -b CUDA0 -p 'type_a=q4_0,type_b=f32,n_mats=256,n_used=16,b=0,m=640,n=16,k=2560,'`
     -> illegal memory access,
  4. apply `tasks/mmq-amend-29953.patch` and run it again -> passes. Then repeat with `guard:ids_dst`.

## Measured on gfx1151 (MaxusAI/ollama#449)

The ROCm host answered the ask. **The amendment is confirmed on hardware, and four things here were wrong.**
Full reply on the issue; build was a Ryzen AI Max+ 395 / Radeon 8060S, ROCm 7.2.1, b11081 and `3070d927f`.

- **hipcc's tables are byte-identical to nvcc's**, for b11081 and `dd266785c` alike, and `mmq-jpad-cost` matches.
  So the modelled gfx1151 column was right about the tables, and is no longer modelled.
- **The gap reproduces, and the amendment fixes it.** Under `guard:src1` on q2_K at `J = 80` with one row in the
  last expert: widest-tile 903 aborts 6/6, amended passes 6/6, every passing run matching the CPU backend.
- **The guard allocator ports to HIP**, which #449 wrongly told them not to bother with: HIP VMM is supported on
  that APU, granularity 4 KiB, and `vendors/hip.h` already maps the `cuMem*` calls. Their three edits
  (`cuCtxGetDevice` -> `ggml_cuda_get_device()`, the granularity enumerator, a `char *` cast on `b.base`) are now
  in [tasks/mmq-debug-alloc.cuh](tasks/mmq-debug-alloc.cuh) behind `GGML_USE_HIP`.
- **Counted is not reachable.** Of the 25,384 shapes the check calls short for the widest-tile rule, **2,648 can
  reach MMQ** on that device and **2,616** have a routing that reads past the allocation: q2_K on RDNA3 only
  takes MMQ with at least 64 experts or at most 128 tokens, which rules out both worst cases the check names,
  including all 128 one-expert-per-token shapes and all 45 dense ones. "Short in N shapes" in this file means
  *the rule cannot be shown sufficient for N shapes*, not *N shapes fault*.
- **`test_mul_mat_id` cannot produce the gfx1151 gap at all.** Its routing is uniform, so at the ~65 rows per
  expert that `J = 80` needs, the last expert holds ~65 rows and the read stays inside the padding. The ROCm host
  wrote a program that builds the routing by hand, putting token 0 alone in the last expert. **That is a limit of
  the harness in this file**, not of the defect: the same uniformity is why `ids64` never faults on src1 here.
- **A gap in main's 903 that this file missed:** `ne12*n_expert_used` is short on gfx1151 for a 6-of-128 K-quant
  MoE at 5 tokens (q4_K, `J = 16`, `nthreads = 128`, need 17, `get_J_max(30) = 16`), aborting 2/2 under
  `guard:src1`. Production's nemotron3 is 6-of-128 but its experts are q5_0/q8_0, which stay on MMVQ to 8 tokens.
- **#29953's head is clean on gfx1151 too**: 72/72 on the twelve cases and 36/36 on the hand-routed ones, with
  the device reporting src1 padded by the padded tile and `ids_dst` by `J_best-1`. Once the pin includes it,
  903's src1 and `ids_dst` padding is redundant on ROCm as well as CUDA.
- **The retracted NVFP4 claim is CUDA-only in any case**, since `use_native_fp4` needs
  `blackwell_mma_available()`.
- Production on gfx1151 is unaffected under all three forms of 903, with a tightest margin of 144 bytes under
  main's rule. Its MoE experts are q4_K/q6_K/q8_0 (`qwen3.6:35b-a3b`, `qwen3-vl:30b-a3b`), q4_K/q5_0/q8_0
  (`gemma4:26b-a4b`) and q5_0/q8_0 (`nemotron3:33b`), read from that host's GGUF headers.

## For the fork

- **Compat 903 needs the same amendment.** As merged in #448 it pads by the widest tile, which covers every
  NVIDIA shape checked but is short by up to 5 blocks in 25,384 gfx1151 shapes (q2_K and q3_K, where `J = 80`
  with `nthreads = 256` needs 85 blocks and the widest tile gives 80). The ROCm host serves gfx1151, so this is
  not hypothetical for the fork even though it is not a shape production's models reach. Amended on 2026-10-05.
  - **The gfx1151 column is modelled, not measured**, and that is asked of the ROCm host in
    MaxusAI/ollama#449: those counts come from `nvcc` compiling the host-side AMD branch of
    `ggml_cuda_mmq_get_config()`, so if hipcc resolves the config tables differently the shortfall could be
    larger, smaller or absent. Until that comes back, treat the gfx1151 numbers here as a prediction.
- **Nothing production serves is affected today.** Its MoE GGUFs use 8 of 128 experts (gemma4:26b-a4b), 8 of 256
  (qwen3.6:35b-a3b) and 6 of 128 plus a shared one (nemotron3:33b), all with `J = 128` at the image ubatch sizes
  it runs, where `T == J*B` and #29941's padding already covers src1. The `ids_dst` read happens on every MoE
  call but lands in `expert_bounds`.
- **Re-cutting 903 against the amendment is the maintainer's call.**
