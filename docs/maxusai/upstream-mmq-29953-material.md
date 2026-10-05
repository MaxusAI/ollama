# llama.cpp#29953 and the two gaps it leaves: facts to write from

Companion to [upstream-mmq-successor-material.md](upstream-mmq-successor-material.md). That file was written
against master `dd266785c` (#29941 merged) and argued for padding by the widest tile. This file supersedes its
arithmetic: **the required src1 padding is larger than every rule proposed so far assumed, including ours**, and
#29953 as first published was the right shape of fix but short in two places. **Its head, `3070d927f`, fixes
both** -- see the Conclusion below. Elsewhere in this file, "#29953" without "head" means the published version,
`5bd8b0013`.

> [!IMPORTANT]
> **This is raw material, not text to post.** ggml-org/llama.cpp prohibits AI-written posts (bug reports, pull
> request descriptions, replies, ...), and undisclosed AI use can get the account banned. The maintainer writes
> the reply and any PR description by hand, from the facts below. The code here (the amendment, the test cases,
> the checks, the harness) is AI-generated, which the project allows **with disclosure**: fill in the template's
> `AI usage disclosure:` line.

## Conclusion

**#29953's head, `3070d927f`, is correct, and nothing about its correctness is owed upstream.** It chooses
`J_best` once, on the host, before allocating; launches exactly that tile (`mul_mat_q_switch_J` now switches on
`args.J_best`); pads src1 by that config's padded y tile, `GGML_PAD(J_best*sizeof(block_q8_1_mmq),
nthreads*sizeof(int))` bytes, which is the load loop's own extent; and pads `ids_dst` by `J_best - 1`. The padding
is computed from the config that is launched, so no shape can outgrow it.

Measured on the exact head, on four architectures:

| where | by | how | result |
|---|---|---|---|
| sm_120, RTX PRO 6000 Blackwell | this host | 13 cases x stock, three guard-page modes and exact-size memcheck, the published #29953 alongside as the positive control | **190/190 runs pass**, no abort, 0 memcheck errors; the control fails in every mode it ran in |
| sm_75, RTX 2080 Ti | this host | `ids16` and `dense321` x stock and two guard-page modes, same control | **18/18**; the control aborts every run that can fault |
| GB10, sm_121a | hclsys, on #29953 | exact-size `cudaMalloc` per buffer under memcheck `--padding 65536`: 16 routing seeds at `n = 100`, a sweep of 48 shapes at 2 seeds, and `test-backend-ops` | 0 errors: 16/16 seeds, where the published version failed 10/16; 96/96 runs; 2291/2291 |
| gfx1151, Radeon 8060S | the ROCm host, #449 | the twelve cases and the hand-routed shapes, the device reporting the head's padding | 72/72 and 36/36 runs |

hclsys tested `20c31408`, and all of `ggml/src/ggml-cuda` is byte-identical between that and the head, so it is the
head's result. The CPU sweep with the head's rule added is covered on all ten architectures, by construction (see
the note under "Ten architectures"). Details: "The exact head, measured".

**What upstream still lacks is a guard.** The head computes the y tile's size twice -- in
`mmq_get_nbytes_shared()` and in its new padding line, spelled differently -- and those two drifting apart is what
this bug was. On the stock pool nothing would notice it happen again. "A guard upstream could take" has a patch on
the head (+70/-9 lines): one helper for both, and a compile-time check over all ten config tables, verified with
nvcc and, by the ROCm host, with hipcc. Whether to offer it is the maintainer's call.

**For the fork: keep compat 903 until the pin includes #29953**, then retire its padding by the gate under "For
the fork" and keep the guard.

## Upstream got there first: #29953's head carries both amendments (pushed 2026-10-04 18:54Z)

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

hclsys's re-test was of `20c31408`, not of the head. The head is a squash of `5bd8b001` and `20c31408` onto
`dd266785`, and all of `ggml/src/ggml-cuda` is byte-identical between `20c31408` and `3070d927f` (checked
2026-10-05), so their GB10 result is the head's result.

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

## Where it stood: #29953 as published (`5bd8b0013`)

Kept as the record of what the published version did. Its head fixes both gaps below.

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
| #29953 as published (`5bd8b0013`) | the launched `J_best`, in blocks | no | no |
| #27044 (the fork's 903 before #448) | `get_J_max(ne12*n_expert_used)` | no | no |
| #448 as published | `get_J_max(type, fallback, cc, 512)` -- the widest tile | yes | yes |
| **#448 amended** (the fork's 903 now) | the widest **padded** tile, `max_J ceil(T/B)` | yes | yes |
| **#29953 head** (`3070d927f`) | the launched config's padded tile, `T` bytes | yes, `J_best - 1` | no, and none needed (retracted above) |

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
| **#29953 as published** | **76,640 (worst 6)** | **1,531,586 (worst 6)** | **all 4,717,328** | **90 (worst 5)** |
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
program's output; on sm_120 #29953 as published is short in 4,488 one-expert and 134,640 multi-expert shapes.
gfx1151's column comes from the host-side config tables as nvcc compiles them, not from a measurement on AMD
hardware.

A count here is **shapes where the rule can be short**, not shapes that fault. The check asks the question a
padding rule has to answer -- is the allocation large enough for the worst routing this shape admits, one row in
the last non-empty expert's last tile -- and `ne12*n_expert_used` rows over `ne02` experts always admits it.
Whether a particular run reads past depends on its routing, on where the pool put the buffer, and on what the
pool mapped after it.

The head's rule is not in this five-architecture run; it is in the ten-architecture run below, covered.

Note the direction of the trade: #29953 as published is short in **more** shapes than #29941 but by **less** --
at most 6 blocks instead of 63 -- because it fixes the rounding and leaves only the padded-tile term.

## Ten architectures: the amended rule is sufficient by construction

The amended rule takes the **maximum padded tile over every config that exists** for `(type, fallback, cc)`. The
launch can only pick one of those configs, filtered further by shared memory, and `smpbo` only ever *removes*
candidates. So the amended rule is sufficient on any architecture and any shared-memory limit, given only that
`mul_mat_q_switch_J` keeps picking from `ggml_cuda_mmq_get_config(type, J, fallback, cc)` for `J` in 8..128. The
sweeps confirm the construction; they are not what makes it hold.

Re-run over ten architectures -- adding sm_70, sm_80, sm_90, CDNA3 and RDNA4 to the original five, 9,431,816
ids-branch shapes ([tasks/mmq-successor-results/check-rules-wide.txt](tasks/mmq-successor-results/check-rules-wide.txt)):

| rule | src1, >=2 experts/token | worst | short on which architectures |
|---|---|---|---|
| #29941 (`ne12`), master | 1,255,564 | 63 blocks | all ten |
| #29953 as published (`5bd8b0013`) | 3,520,756 | **12 blocks** | all ten |
| #27044, the fork's 903 before #448 | 635,468 | 17 blocks | all ten |
| #448 as published | 626,512 | **7 blocks** | **gfx1151 25,256, CDNA3 576,000, RDNA4 25,256; zero on all seven NVIDIA** |
| **#448 amended** | **covered** | - | none |
| **#29953 head (`3070d927f`)** | **covered** | - | none |

> [!NOTE]
> **Corrected 2026-10-05.** This table used to label the published row "#29953 at `3070d927f`". The checker's #29953
> rule is the published one -- pad by the launched `J_best` in blocks -- and the head's rule was not in it at all.
> Read literally, the old label said upstream's current fix is short in 3.5 million shapes, which is false. The
> head's rule (the launched config's padded tile, and `J_best - 1` for `ids_dst`) is now in
> [tasks/mmq-rules-check.cu](tasks/mmq-rules-check.cu) and is covered in all four tallies on all ten architectures:
> 9,431,816 ids-branch shapes and 203,200 dense ones. That is by construction -- its padding is the load loop's own
> extent for the launched config -- so the sweep shows the model agrees, not that the head is right; the
> measurements in the Conclusion are the evidence for that. Built against `dd266785c` and against `3070d927f`
> itself, the check prints byte-identical output; the config tables are the same files in both.

**This widens the case for the amendment well beyond gfx1151.** `nthreads` is 512 on CDNA, not 256, so the
padded tile is larger: the worst case is CDNA3, q4_0, non-fallback, `J = 64` with `nthreads = 512`, where
`GGML_PAD(64*144, 2048) = 10,240` bytes needs 71 blocks against the widest tile's 64. The dense branch is short
there too (1,960 CDNA3 shapes against 45 each on sm_75, gfx1151 and RDNA4).

> [!WARNING]
> **sm_70, sm_80, sm_90, CDNA3 and RDNA4 are modelled, not measured** -- there is no such hardware here and
> nobody has run them. gfx1151 was modelled too and came back byte-identical on hipcc (#449), which is some
> evidence the modelling is sound, but it is not proof for CDNA or RDNA4. Their `smpbo` values are the documented
> per-architecture limits rather than readings, which the construction above makes harmless. And as #449
> established, a counted shape is not a reachable one: these are counts.

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

| `MMQ_DEBUG_ALLOC` | what it exposes | `ne11` | `ne12` | #29953 as published | #27044 | #448 |
|---|---|---|---|---|---|---|
| unset (stock pool) | nothing -- the pool hides both | pass | pass | pass | pass | pass |
| `guard:src1` | the padded-tile term | abort | abort | **abort** | pass | pass |
| `guard:ids_dst` | the unpadded `ids_dst` | abort | abort | **abort** | abort | pass |

`#27044` passes `guard:src1` here only because `get_J_max(ne12*n_expert_used) = get_J_max(256) = 128`, far more
than the 21 blocks this shape needs; it is short elsewhere (see the CPU check). At #29953's head both guarded
rows pass: see "The exact head, measured".

**Why this shape.** `need` is the worst case over all routings. What a given case actually reads past the data
is `ceil(T/B) - r` blocks, where `r` is the last non-empty expert's row count, so the worst case needs `r = 1`.
Whether a case can realise it is set by the mean rows per expert, `ne12*n_expert_used / ne02`:

| case | rows | experts | mean rows/expert | `J` | realises the worst case? |
|---|---|---|---|---|---|
| `ids16` | 256 | 256 | 1.0 | 16 | yes, every run |
| `j100_b0`, `j100_b1` | 1,000 | 512 | 2.0 | 112 | sometimes |
| `t100` | 800 | 256 | 3.1 | 112 | sometimes |
| `ids64` | 6,400 | 256 | 25.0 | 112 | no: reads about 89 blocks past, which `ne12`'s 96 covers |

So `ids16` is the case to hand over: one row per expert, every run, and a 5-block shortfall against the published
#29953 rather than one. `ids64` is the control that shows the routing matters -- it passes `guard:src1` under
every rule but `ne11`. An over-read is a function of the routing as well as the shape, which is the third reason
this bug is hard to hit on purpose, after the pool and the fixup buffer.

## The amendment

[tasks/mmq-amend-29953.patch](tasks/mmq-amend-29953.patch) applies on top of #29953 (verified against
`5bd8b0013`), 8 changed lines, `mmq.cu` only:

- Carry the chosen config's thread count out of the selection loop as
  `nbytes_pad_y = GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int))`, and pad src1 by that
  instead of `J_best * sizeof(block_q8_1_mmq)`, in both branches. It is the expression
  `mmq_get_nbytes_shared()` already uses for the tile the load copies into.
- `ids_dst` gets `J_best` more entries.
- **Retracted, see the top of this file** -- kept so the patch can be read: for NVFP4 with native FP4,
  `src1_scale` gets `J_best` more entries in both branches: the stream-k fixup
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
tested change. **Since built and verified against the head**, with a compile-time check added: see "A guard
upstream could take".

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

- **CPU only, nvcc and no GPU**, from a llama.cpp checkout at `dd266785c` or later, #29953's head included:
  `tasks/build-check.sh <checkout> tasks/mmq-rules-check.cu mmq-rules-check && <checkout>/mmq-rules-check`.
  The check carries its own copy of `ggml_cuda_mmq_get_J_max()`, which #29953 deletes; built against
  `dd266785c` and against `3070d927f`, it prints output byte-identical to the build that used the library's
  copy -- every rule's counts over all 9.4 million shapes.
  `build-check.sh` compiles from inside the checkout, so a relative output path lands there. The exit code is
  non-zero only if 903's rule or #29953's head leaves a shape short. `--design` lists shapes where only
  `ids_dst` is short.
- **GPU:** `tasks/mmq-rules-gpu.sh <llama.cpp checkout at dd266785c> [out dir]`. It applies the test cases,
  builds each rule twice (with and without the debug allocator) via
  [tasks/mmq-variant.py](tasks/mmq-variant.py), and runs every case stock, under `guard:ids_dst`, under
  `guard:src1`, and under memcheck with exact-size allocations. It is resumable.
- **The shortest path for a reviewer**, on any NVIDIA GPU, from a `dd266785c` checkout with
  `tasks/mmq-tests.patch` applied:
  1. `mmq-variant.py <checkout> p29953 --debug` -- #29953 as published, plus the debug allocator -- and build
     `test-backend-ops`;
  2. `MMQ_DEBUG_ALLOC=guard:src1 test-backend-ops test -o MUL_MAT_ID -b CUDA0 -p 'type_a=q4_0,type_b=f32,n_mats=256,n_used=16,b=0,m=640,n=16,k=2560,'`
     -> illegal memory access;
  3. `mmq-variant.py <checkout> head29953 --debug` -- #29953's head, byte-identical to `3070d927f` plus the
     allocator -- rebuild, and run it again -> passes. Then repeat both with `guard:ids_dst`.

  [tasks/mmq-rules-head29953.sh](tasks/mmq-rules-head29953.sh) runs this and the rest of "The exact head,
  measured".
- **The guard**, no GPU: [tasks/mmq-29953-guard-verify.sh](tasks/mmq-29953-guard-verify.sh), next to a
  llama.cpp clone that has `3070d927f`.

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

## The exact head, measured

`3070d927f` itself, not an equivalent. On a `dd266785c` checkout, `mmq-variant.py`'s `head29953` swaps in the
head's `mmq.cu` and `mmq.cuh` -- the only two files the head commit touches -- after checking that the checkout is
the head's parent and that the commit touches nothing else; without `--debug` the result is byte-identical to
`3070d927f`. Same harness as "Results", run by
[tasks/mmq-rules-head29953.sh](tasks/mmq-rules-head29953.sh), with the published #29953 alongside as the positive
control: a clean head means something only where the same harness shows the published version failing. The
controls for the combined guard, `guard:ids_dst` and memcheck were a second pass,
[tasks/mmq-rules-head29953-controls.sh](tasks/mmq-rules-head29953-controls.sh).

Totals, from `mmq-rules-table.py --summary` ("aborted" is a process that died; memcheck errors are summed over the
runs made under the sanitizer):

**sm_120**, RTX PRO 6000 Blackwell, CUDA 12.8, every case in [tasks/mmq-tests.patch](tasks/mmq-tests.patch):

| rule | mode | cases | runs | passed | aborted | memcheck errors |
|---|---|---|---|---|---|---|
| #29953 as published | exact | 2 | 2 | 0 | 0 | 13183 |
| #29953 as published | guard | 2 | 6 | 0 | 6 | - |
| #29953 as published | guard_ids_dst | 2 | 6 | 0 | 6 | - |
| #29953 as published | guard_src1 | 3 | 30 | 12 | 18 | - |
| #29953 head (`3070d927f`) | stock | 13 | 39 | 39 | 0 | - |
| #29953 head (`3070d927f`) | exact | 13 | 13 | 13 | 0 | 0 |
| #29953 head (`3070d927f`) | guard | 13 | 39 | 39 | 0 | - |
| #29953 head (`3070d927f`) | guard_ids_dst | 13 | 39 | 39 | 0 | - |
| #29953 head (`3070d927f`) | guard_src1 | 13 | 60 | 60 | 0 | - |

**sm_75**, RTX 2080 Ti, CUDA 12.8: `ids16`, and `dense321`, the plain `MUL_MAT` that the published version
regressed against master:

| rule | mode | cases | runs | passed | aborted | memcheck errors |
|---|---|---|---|---|---|---|
| #29953 as published | guard_ids_dst | 2 | 6 | 3 | 3 | - |
| #29953 as published | guard_src1 | 2 | 6 | 0 | 6 | - |
| #29953 head (`3070d927f`) | stock | 2 | 6 | 6 | 0 | - |
| #29953 head (`3070d927f`) | guard_ids_dst | 2 | 6 | 6 | 0 | - |
| #29953 head (`3070d927f`) | guard_src1 | 2 | 6 | 6 | 0 | - |

- **The head passes everything**: 190 runs on sm_120 across 13 cases and five modes, and 18 on sm_75, with no
  abort and no memcheck error.
- **The control fails in every mode, on both cards**: `ids16` aborts in every guarded run of the published
  version; under memcheck `one113` reports 12,885 errors and `ids16` 298; `dense321` aborts 3/3 under
  `guard:src1` on sm_75. The three sm_75 control runs that pass are `dense321` under `guard:ids_dst`, which has
  no `ids_dst` to guard.
- **The one-block cases are a rate, as before**: under `guard:src1` the published version aborted `j100_b1` 3
  times in 10 and `j100_b0` 5 times in 10 (4 and 1 in the earlier run), the head 0 in 10 on both.
- Per-case tables: [matrix-head29953-sm120.md](tasks/mmq-successor-results/matrix-head29953-sm120.md) and
  [matrix-head29953-sm75.md](tasks/mmq-successor-results/matrix-head29953-sm75.md), raw rows beside them.

## A guard upstream could take

The head's fix is right, but it leaves the y tile computed twice: `mmq_get_nbytes_shared()` sizes the
shared-memory tile with `GGML_PAD(nbs_y, config.nthreads*sizeof(int))`, and `ggml_cuda_mul_mat_q` sizes the global
src1 padding with its own ceiling-division expression. Those two drifting apart is what this bug was, and nothing
would see it happen again: on the stock pool the over-read lands in mapped memory and `test-backend-ops` passes.

[tasks/mmq-29953-y-tile-guard.patch](tasks/mmq-29953-y-tile-guard.patch) applies to `3070d927f` (70 lines added,
9 removed, `mmq.cu` and `mmq.cuh`):

- **One helper.** `ggml_cuda_mmq_get_nbytes_y_tile(config)` in `mmq.cuh` returns
  `GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int))`. `mmq_get_nbytes_shared()` calls it
  for its y term, and the selection loop carries `nbytes_y_tile_best` out instead of `nthreads_best`, so the
  padding *is* the tile. **No runtime value changes:** every config table uses 128, 256 or 512 threads, where
  `GGML_PAD` and the head's division agree.
- **A compile-time check.** For every config table and type, the helper must cover the load loop's extent,
  written from the loop with exact ceiling division -- `ceil(J*MMQ_TILE_Y_K / nthreads) * nthreads` ints -- not
  from the helper or from `GGML_PAD`. One `static_assert` per (table, type), host pass only: the structure of
  the fork's compat 903 guard (#451). They are gathered by an explicitly instantiated class that inherits every
  per-type guard, the ROCm host's variant (#455, #456). The first form gathered them with a `static_assert` on a
  fold, which under clang turned non-constant once a per-type assertion failed and added one error per failing
  table, worded like a constant-evaluation limit.

Verified with nvcc 12.8 here and with hipcc on the ROCm host. nvcc, for sm_120, by
[tasks/mmq-29953-guard-verify.sh](tasks/mmq-29953-guard-verify.sh), which scores each build by why it failed -- every error must be the guard's and the tables it names must be the expected
ones -- not just by its exit code
([tasks/mmq-successor-results/verify-29953-guard.txt](tasks/mmq-successor-results/verify-29953-guard.txt)):

```
host load at start (1/5/15 min): 38.78 35.94 28.60, 32 cores
1-head-unpatched                   rc=0   ok   want=ok   as-expected    13.5s
patch applies to a pristine 3070d927f6c172242d56a2c1fb897e838e7b2632
2-head-with-guard                  rc=0   ok   want=ok   as-expected    19.1s
3-helper-shrunk-to-J-blocks        rc=2   fail want=fail as-expected     5.0s
    errors=220, from the guard=220, tables named: ampere blackwell cdna gcn pascal_dp4a pascal_older rdna2 rdna3 rdna3_5 rdna4
4-helper-rounded-to-one-warp       rc=2   fail want=fail as-expected     5.0s
    errors=220, from the guard=220, tables named: ampere blackwell cdna gcn pascal_dp4a pascal_older rdna2 rdna3 rdna3_5 rdna4
mutation 5 rewrote:     CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true); ->     CASE(GGML_TYPE_MXFP4, 192, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true);
5-blackwell-config-at-192-threads  rc=2   fail want=fail as-expected     6.1s
    errors=1, from the guard=1, tables named: blackwell
mutation 5b rewrote:     CASE(GGML_TYPE_Q1_0, 256, 2, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true); ->     CASE(GGML_TYPE_Q1_0, 160, 2, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true);
5b-rdna3_5-config-at-160-threads   rc=2   fail want=fail as-expected     6.0s
    errors=1, from the guard=1, tables named: rdna3_5
6-guard-compiled-out               rc=0   ok   want=ok   as-expected    14.2s
VERIFY29953GUARD COMPLETE 2026-10-05T04:51:06Z all_as_expected=1
```

- **3** puts the published #29953's padding, `J` blocks, back into the helper: every table fails, 220 (table,
  type) pairs, and nothing else does.
- **4** is a plausible wrong simplification, rounding to one warp's ints: the same.
- **5** is what the exact arithmetic is for. `GGML_PAD` is a bit mask that is only right for power-of-two
  granularities, the config tables only assert `nthreads % 32 == 0`, and the kernel places `tile_x` with the
  same `GGML_PAD`. A 192-thread config would compile today and mis-size the tile; with the guard, the build fails
  and names that table only.
- **5b** is 5 on an AMD table: rdna3_5 at 160 threads. Not 192: rdna3_5's first 256-thread line has `J = 64`, and
  64 x 36 ints is a multiple of 192, so the bit mask happens to round correctly there. The first instruction to
  the ROCm host said 192 and could not fire; they found 160 (#455).
- **hipcc agrees on every build** (ROCm 7.2.1, AMD clang 22, gfx1151, run by the ROCm host:
  [#456](https://github.com/MaxusAI/ollama/pull/456), `rocm-gfx1151/README.md`). The patch compiles; 3 and 4 fail
  with 220 errors, all the guard's, naming all ten tables; blackwell at 192 and rdna3_5 at 160 each name only
  their table; compiled out, it compiles. clang stops at 20 errors unless given `-ferror-limit=0`.
- **No generated code changes, on either compiler.** On hipcc `mmq.o` is byte-identical with and without the
  guard: clang derives `__hip_cuid_*` from the path and the command line. On nvcc whole objects differ between
  two builds of the same source, so the comparison is of the host disassembly, 56,837 lines, identical across the
  first form, the variant, no guard and a repeat build.
- **Cost:** about 3 s per compile of `mmq.cu` on nvcc and 0.8 s on hipcc, paid once per build because the guard
  runs in the host pass only, however many architectures are targeted. nvcc: a median of 15.2 s with the guard
  against 11.8 s compiled out, over five interleaved repeats at a load of 28-36 on 32 cores
  ([tasks/mmq-29953-guard-timing.sh](tasks/mmq-29953-guard-timing.sh),
  [output](tasks/mmq-successor-results/guard-timing.txt)); the first form measured 11.1 against 8.2 s at a load
  of 7-8. hipcc, measured by the ROCm host: 2.89 against 2.06 s, median of five.
- nvcc's front end stops at 100 errors by default, which on the first run cut mutation 3's list to the first five
  tables; the script passes `-Xcudafe --error_limit=100000` so every failing table is named.

**Limits.** The HIP branch is verified by the ROCm host; MUSA is untested, and excluded as in 903 because nothing
here builds it. HIP keys off `__HIP_DEVICE_COMPILE__` because `vendors/hip.h` defines `__CUDA_ARCH__` in every HIP
pass -- the trap the ROCm host caught in #451's first draft. The guard
cannot see call sites: a later change that pads src1 by something other than the helper would pass it. `ids_dst`
needs no guard of this kind; its `J_best - 1` depends on nothing but `J`.

**This is material, not a submission.** Whether to offer it upstream is the maintainer's call, and anything posted
to ggml-org is written by hand.

## For the fork

- **Compat 903 stays as merged in #451** -- the widest padded tile, one helper for the y tile, and the
  compile-time guard -- until the pin includes #29953. It is not replaced by a backport of #29953:
  - the pin, b11081, predates the `prec_src1` refactor (`mmq.cuh` differs from #29953's base `dd266785c` by 230
    changed lines), so a backport would be an adaptation, not upstream's code, and would still be dropped or
    re-cut at the pin move;
  - #29953 is not merged (open, review required, last changed 2026-10-04 18:55Z) and can still change;
  - 903 is measured on sm_120 and gfx1151 and guarded at compile time, and the head has no guard;
  - nothing is waiting on it: even v0.35.1's pin, b11232, predates #29941.
- **The two fixes agree on what matters.** Both pad src1 by a padded y tile and both pad `ids_dst`. 903 pads for
  the widest tile any launch could pick, the head for the tile it launches, so 903 reserves up to 16 KiB more src1
  per call on sm_120 (18,432 bytes against the head's 2,048 at `J = 8`) and a few hundred bytes more `ids_dst`.
- **Retiring 903**, at the first pin move that includes #29953's merge commit:
  1. the patch series, without 903, applies to the new pin with plain `git apply`;
  2. the head's rule is covered in `mmq-rules-check.cu` built against the new pin with `build-check.sh` (it
     already builds against #29953's head unchanged: it carries its own copy of the `get_J_max()` that #29953
     deletes);
  3. `ids16` under `guard:src1` and `guard:ids_dst` on sm_120, and `dense321` under `guard:src1` on sm_75, pass
     on the new pin, with the padding set back to `J_best` blocks as the positive control in the same harness
     (`mmq-variant.py`'s anchors will need moving with the pin);

  then 903's padding goes. **The guard stays**, re-cut guard-only, unless upstream has taken one -- that was
  already the compat README's plan, and [tasks/mmq-29953-y-tile-guard.patch](tasks/mmq-29953-y-tile-guard.patch)
  is its form on the head. If #29953 changes before it merges, re-run "The exact head, measured" against what
  merged.
- **gfx1151 is measured, not modelled** (#449): the widest-tile rule's gap reproduced, aborting 6/6, and the
  amended rule fixed it, passing 6/6, every passing run matching the CPU backend.
- **Nothing production serves is affected today.** Its MoE GGUFs use 8 of 128 experts (gemma4:26b-a4b), 8 of 256
  (qwen3.6:35b-a3b) and 6 of 128 plus a shared one (nemotron3:33b), all with `J = 128` at the image ubatch sizes
  it runs, where `T == J*B` and #29941's padding already covers src1. The `ids_dst` read happens on every MoE
  call but lands in `expert_bounds`.
