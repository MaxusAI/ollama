# A successor to llama.cpp#27044: facts to write from

Companion to [upstream-mmq-submission-material.md](upstream-mmq-submission-material.md), which holds the history up
to #29941's merge. This file holds what a successor PR needs: the defect that is left, the change, the test cases,
and the measurements behind each claim.

> [!IMPORTANT]
> **This is raw material, not text to post.** ggml-org/llama.cpp prohibits AI-written posts (bug reports, pull
> request descriptions, replies, ...), and undisclosed AI use can get the account banned. The maintainer writes the
> PR description, the commit message and every reply by hand, from the facts below. The code here (the change, the
> test cases, the checks) is AI-generated, which the project allows **with disclosure**: fill in the template's
> `AI usage disclosure:` line. See "What happened when we filed it" in the companion file.

## Where it stands (2026-10-04)

- **#29941 merged** at 12:10 UTC as `dd266785c`. It is a single change: the ids branch of `ggml_cuda_mul_mat_q` pads
  `get_J_max(ne12)` blocks instead of `get_J_max(ne11)`. #27044 (`ne12*n_expert_used`) was closed at 12:12, "in favor
  of the other one, assuming the issue is now fixed".
- **The maintainer then posted #445's sm_120 table on #27044** (12:21). The upstream CUDA maintainer replied at 12:47
  that they see no `compute-sanitizer` errors with #29847's shape at 100 tokens, `b` false and true.
- **That matches our own runs.** The table's errors were measured with the src1 buffer in its own exact-size
  `cudaMalloc` (the `MMQ445_EXACT` debug switch). With the stock memory pool, #445 also measured 0 errors for #29941 on
  that shape. The pool maps memory beyond the buffer, so a read past the buffer is still a read of valid memory, and
  memcheck has nothing to report.
- **What is new here:** a `test-backend-ops` case that **aborts on master `dd266785c` as shipped**, with the stock pool,
  without a sanitizer and without any debug code. See "The case that aborts on master".

## The defect that is left

Read at `dd266785c`, `ggml/src/ggml-cuda/mmq.cuh` and `mmq.cu`:

- **Tiles load without a bound.** A tile loads `J` columns of src1, `tile_y[l] = by0[l]` for
  `l < J*MMQ_TILE_Y_K`, and `J` entries of `ids_dst`, `ids_dst_shared[j] = ids_dst[col_low + jt*J + j]` for `j < J`.
  Neither checks the column count. Only the write-back does (`if (j > j_max) return;`).
- **So the last tile reads past the data.** src1 is laid out `[k-plane][column]`. In the last k-plane, a tile holding
  `c` valid columns reads `J - c` blocks of `block_q8_1_mmq` (144 bytes each) past the data, and `ids_dst` `J - c`
  entries past its end.
  - In the ids branch, the last expert's last tile can hold a single row, whatever the batch. So src1 needs at least
    `J - 1` blocks of padding, and `ids_dst` `J - 1` entries.
  - In the dense branch, the last tile holds `ne11 - (ntiles - 1)*J` columns, so it needs `ntiles*J - ne11` blocks.
- **`J` and the padding round in opposite directions.**
  - `mul_mat_q_switch_J` picks the smallest `J` (8 to 128 in steps of 8, a valid config that fits shared memory) that
    gives the fewest tiles for `ncols_opt`. That rounds the batch **up** to a tile.
  - `ggml_cuda_mmq_get_J_max(n)` returns the largest valid config at or below `n`, after rounding `n` **down** to a
    multiple of 8.
  - So below 128 tokens `get_J_max(ne12)` is often smaller than the launched `J`; the CPU check counts the shapes.
    Two examples on sm_120:
    - 120 tokens with 576 rows (`ne01 % 128 != 0`): the fallback configs are `J` = 8, 16, 32, 64 and 128. The launch
      is `J = 128`, and #29941 pads 64 blocks.
    - 100 tokens with 640 rows: there is no config at 104, so the launch is `J = 112`, against a padding of 96 blocks.
- **`ne12*n_expert_used` (#27044) does not close it either.**
  - On NVIDIA it covers the src1 shortfall whenever a token uses two or more experts, because the argument is then
    at least twice the batch.
  - With one expert per token it equals `ne12`, so it is short exactly where #29941 is.
  - On RDNA3 (gfx1151, CPU check only) it is short in 192 shapes, all with 2 or 3 experts per token and 5 to 7
    tokens, where its padding rounds to 0.
  - It never pads `ids_dst`.
- **One more buffer is read per whole tile, for NVFP4 with native FP4:** the y scales, in the stream-k fixup
  write-back (see "The change").
- **Before #24127 (b9992) both branches padded by the widest tile**, `get_mmq_x_max_host(cc)` blocks: 128 on Turing
  and newer. #24127 replaced that with `get_J_max(ne11)`; see
  [mmq-padding-regression-window.md](mmq-padding-regression-window.md).

## The CPU check: every rule, 4.7 million shapes

[tasks/mmq-padding-check.cu](tasks/mmq-padding-check.cu) replays the tile selection against llama.cpp's own config
functions, with no GPU. It covers:
- sm_75, sm_86, sm_89, sm_120 and gfx1151;
- ten quantization types, fallback 0 and 1;
- every MMQ batch up to 1024 tokens, starting above each type's MMVQ limit;
- `n_expert_used` 1 to 16 and broadcast 0 and 1, and on gfx1151 also 8, 32, 128 and 256 experts, since RDNA3 picks
  `J` against the tokens per expert.

Output on master `dd266785c`, nvcc 13.0, 2 minutes:

```
ids branch: 4717824 shapes
 src1_q8_1, one expert per token (n_expert_used = 1):
  #24127 (ne11)                 SHORT in 162688 shapes, worst 127 blocks
  master = #29941 (ne12)        SHORT in 9604 shapes, worst 63 blocks (sm_75, q4_0, fallback=1, 65 tokens: J=128, padding=64)
  #27044 (ne12*n_expert_used)   SHORT in 9604 shapes, worst 63 blocks (the same shapes)
  successor (widest tile)       covered
 src1_q8_1, two or more experts per token:
  #24127 (ne11)                 SHORT in 4546296 shapes, worst 127 blocks
  master = #29941 (ne12)        SHORT in 294826 shapes, worst 63 blocks (sm_75, q4_0, fallback=1, 65 tokens, 2 used: J=128, padding=64)
  #27044 (ne12*n_expert_used)   SHORT in 192 shapes, worst 15 blocks, all on gfx1151 (q4_K, fallback=0, 5 tokens, 2 used, 8 experts: J=16, padding=0)
  successor (widest tile)       covered
 ids_dst, any n_expert_used:
  every rule but the successor  SHORT in all 4717824 shapes, worst 127 int32 entries
  successor (widest tile)       covered
dense branch: 101600 shapes
  master (ne11)                 SHORT in 140 shapes, worst 7 blocks, all on gfx1151 (q4_0, fallback=0, ne11=9: J=16, padding=0)
  widest tile                   covered
```

The per-architecture counts are in the program's output. On NVIDIA, the dense branch is covered by master's rule.
gfx1151's counts come from the host-side config tables as nvcc compiles them, not from a measurement on AMD hardware.

## The change

[tasks/mmq-successor-fix.patch](tasks/mmq-successor-fix.patch), in `mmq.cu` only:
- `J_pad = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, 512)`: the widest tile that has a config. It is at least
  every `J` that `mul_mat_q_switch_J` can launch, since the switch launches only valid configs of at most 128.
- Both branches pad src1 by `J_pad` blocks. `ids_dst` gets `J_pad` more entries.
- **NVFP4 with native FP4 only:** `src1_scale` gets `J_pad` more entries in both branches.
  - The stream-k fixup write-back calls `write_back(..., y_scale, I, I, J)`, so it reads `y_scale[j]` for all `J`
    columns of its tile, and the last tile can read up to `J - 1` scales past the end.
  - This was found by reading the code, not measured. The read happens only on the path that allocates the fixup
    buffer right after `src1_scale`, so the pool always hides it.
  - The GPU runs here use the q8_1 path: an sm_120 build without `120a` has no native FP4.
- **The cost:** on NVIDIA `J_pad` is 128. That is 18,432 bytes of src1 padding, 512 bytes of `ids_dst` and, for
  native FP4, 512 bytes of scales per call, whatever the batch. The src1 padding is what it was before #24127.
- **The extra entries are read but never used.** The padding is never initialised. `ids_dst`'s extra entries reach
  shared memory, but both write-back paths (`mmq_write_back_*` and `mul_mat_q_stream_k_fixup`) stop at `j > j_max`,
  with `j_max = col_diff - jt*J - 1`. The columns that the src1 padding feeds are discarded the same way.

## The test cases

[tasks/mmq-successor-tests.patch](tasks/mmq-successor-tests.patch) adds ten `test_mul_mat_id` cases to
`make_test_cases_eval()`, after the "more than 256 experts" block:

| case | shape (type, experts, used, b, rows, tokens, k) | what it shows |
|---|---|---|
| `p29847_b0`, `p29847_b1` | q4_0, 512, 10, false/true, 640, 508, 2560 | #29847's reproducer: aborts without padding, fixed by #29941 |
| `j100_b0`, `j100_b1` | q4_0, 512, 10, false/true, 640, 100, 2560 | the shape the upstream CUDA maintainer ran: short under #29941, hidden by the pool |
| `e120_b0`, `e120_b1` | q4_K, 256, 10, false/true, 576, 120, 1536 | **aborts on master `dd266785c`**: see the next section |
| `one113` | q4_K, 64, 1, false, 576, 113, 16384 | one expert per token: short under #29941 and #27044 alike |
| `orig2040` | q4_K, 256, 8, true, 512, 2040, 2048 | #27044's original fault |
| `fb65`, `t100` | q4_K, 256, 8, true, 576/512, 65/100, 2048 | #445's short-batch cases |

## The case that aborts on master

**Measured.** `test-backend-ops test -o MUL_MAT_ID -p 'type_a=q4_K,type_b=f32,n_mats=256,n_used=10,b=0,m=576,n=120,k=1536,'`
on master `dd266785c` with only the test cases added: "CUDA error: an illegal memory access was encountered", **10 of
10 runs** (`b=0` and `b=1`, 5 each). That was the stock pool, no sanitizer, an RTX PRO 6000 Blackwell (188 SMs).
The build here used nvcc 12.8 with CUDA 12.1's runtime; the driver's run with a consistent toolkit is under
"Results".

**Why this shape and not #445's.** Two things must hold for an over-read to leave mapped memory:
- **The buffer must end near the end of the pool's mapping.**
  - The VMM pool bumps allocations from its base, rounds each to 128 bytes, and maps in multiples of the granularity,
    measured at 2 MiB on this card.
  - The ids branch allocates `ids_src1`, `ids_dst`, `expert_bounds` and then `src1_q8_1`. So in a process that runs
    the case alone, src1 ends at `E` and the mapping at `E` rounded up to 2 MiB.
  - For `e120`, `E` is 2,093,696 bytes, 3,456 bytes before the mapping's end. The last tile reads `128 - c` blocks
    past the data, where `c`, the last expert's column count, is about 5. That is `64 - c` blocks, about 8.5 KB, past
    #29941's 64 blocks of padding.
  - For #29847's shape at 100 tokens, the slack is 1,290,112 bytes. That is why the upstream CUDA maintainer sees
    nothing.
- **No stream-k fixup buffer may follow it.**
  - `launch_mul_mat_q` allocates `nsm*J*I` floats after src1 (12,320,768 bytes on 188 SMs) when the config uses
    stream-k and the tiles do not divide into the blocks launched. On NVIDIA that is `ntiles` when they fill at least
    90 % of their waves, and `nsm` otherwise. On sm_120, every config of q4_0, q8_0, q2_K, q3_K, q4_K and q6_K uses
    stream-k.
  - **Our first candidate failed here.** It was 67 tokens, 128 experts and 6 used, ending exactly at 2 MiB, and it
    passed 5 of 5 runs: its 640 tiles fill 85 % of the waves on 188 SMs, so a fixup buffer followed.
  - A logged `cuMemMap` showed the pool mapping 2 MiB and then 12 MiB.
  - `e120` has 1,280 tiles. They fill at least 90 % on 82, 108, 128, 132, 170 and 188 SMs (RTX 3090, A100, RTX 4090,
    H100 SXM, RTX 5090, RTX PRO 6000), so no fixup buffer follows on any of them.

[tasks/mmq-crash-shape.cu](tasks/mmq-crash-shape.cu) is the search that found it, using the same pool and fixup model.

## The over-read on a real model (2026-10-05)

The test-backend-ops cases are synthetic. The over-read also reproduces end to end on a served model, which is how it
was first found (`qwen35moe-mmq-investigation.md`). On `qwen3.6:35b-a3b-q4_K_M` (qwen35moe, 256 experts, 8 used) with
one 3072x1728 image and `num_batch=2048` (the image prefills as 2048+2032-token ubatches through the experts), on the
CUDA host (sm_120, b11081):

| libggml-cuda.so | src1 allocation | num_ctx | outcome |
|---|---|---|---|
| raw (upstream `ne11`) | VMM pool (shipping) | 8192 and 33792 | HTTP 200, image decoded |
| raw (upstream `ne11`) + 910 | exact-size `cudaMalloc` | 8192 | **illegal memory access at `decoding image batch 1/2`, core dump** |
| #29941 (`ne12`, merged) + 910 | exact-size `cudaMalloc` | 8192 | HTTP 200, image decoded cleanly |
| new 903 (widest tile) + 910 | exact-size `cudaMalloc` | 8192 | HTTP 200, image decoded cleanly |

The broadcast gate/up MUL_MAT_ID has `ne11 == 1`, so upstream's `get_J_max(ne11) = 0`: `src1_q8_1` gets no tail
padding and the kernel reads a whole tile past it. The shipping VMM pool maps memory beyond the buffer, so the read
lands in valid memory and the run completes -- the bug is latent, which is why it first looked like an intermittent
crash gated on `num_ctx > 32768`. Debug patch `910-mmq-exact-debug.patch` (env `MMQ_EXACT=1`, not shipped) gives
`src1` its own exact-size allocation so the read crosses the boundary deterministically: the raw kernel then
core-dumps at the image-batch decode (`llama_context::process_ubatch -> decode`, `ggml-cuda.cu:108`), and the
widest-tile 903 decodes it cleanly. Logs and the full table: [tasks/mmq-successor-results/real-model/](tasks/mmq-successor-results/real-model/).

compute-sanitizer cannot instrument the model through `ollama serve`: ollama re-execs its runner with a rebuilt
environment that drops the sanitizer injection, so only the parent is instrumented. The precise
`Invalid __global__ read ... mul_mat_q` line is from the test-backend-ops path under memcheck; on the real model the
exact-size allocation turns the same read into the deterministic crash above. **#29941 does not crash this path:** `get_J_max(ne12)` with `ne12 = 2048` pads the widest tile, so the merged fix
covers the large-image src1 read; its residual gaps (`ids_dst`, and src1 below 128 tokens) are the test-backend-ops
exact rows, not this one.

**What it does not show.** The case aborts when it runs on a pool nothing larger has used before it. That holds for
`-p` on the case alone, and for `-o MUL_MAT_ID` if no earlier case needs more pool. In a full run the pool may
already be larger, and the read lands in mapped memory again: the same holds for #29847's cases. On a GPU whose SM
count is not listed, recompute with `mmq-crash-shape.cu`. Its search found no one-expert-per-token shape that aborts
on all six SM counts with under 800 MiB of weights, so `one113` is evidence only under the exact allocation.

## Results

_Pending: `mmq-successor-gpu.sh`'s end-to-end run (CUDA 12.8 throughout, sm_120) is in progress, and its table
replaces this paragraph._

## Reproduce

- **CPU only, nvcc and no GPU**, from a llama.cpp checkout:
  `nvcc -std=c++17 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda mmq-padding-check.cu -o mmq-padding-check && ./mmq-padding-check`.
  The exit code is non-zero only if the successor's rule leaves a shape short.
- **GPU:** `docs/maxusai/tasks/mmq-successor-gpu.sh <llama.cpp checkout at dd266785c or later> [out dir]`.
  - It applies the test cases.
  - It builds eight `test-backend-ops`: four padding rules (`pre29941`, `master`, `27044`, `successor`), each without
    and with the `MMQ_EXACT` debug switch ([tasks/mmq-successor-variant.py](tasks/mmq-successor-variant.py)).
  - It runs every case three ways: stock without a sanitizer (`REPS` times), stock under memcheck, and under memcheck
    with `ids_dst` and src1 in exact-size allocations.
- **The shortest path for a reviewer:** apply `mmq-successor-tests.patch` to master, build, and run
  `test-backend-ops test -o MUL_MAT_ID -p 'n_mats=256,n_used=10,b=0,m=576,n=120,k=1536,'` on an NVIDIA GPU from the
  list above. Then apply `mmq-successor-fix.patch` and run it again.

## For the fork

- **Compat 903 is this change since 2026-10-05,** on the maintainer's word. Before that, it carried #27044's line.
  It is the same change as `tasks/mmq-successor-fix.patch`, cut against b11081, whose ids branch still reads
  `ne11`. It is marked `MAXUSAI (compat 903)`.
- **The facts behind the switch:** three gaps in #27044's line.
  - The line is short with one expert per token, and on gfx1151 with 2 or 3 experts at 5 to 7 tokens. Neither case is
    a model production serves. Its MoE GGUFs use 8 of 128 experts (gemma4:26b-a4b), 8 of 256 (qwen3.6:35b-a3b) and 6
    of 128 plus a shared one (nemotron3:33b), read from their GGUF headers.
  - It does not pad `ids_dst`. That read happens in every MoE model, production's included, but it lands in the
    pool's next allocations, `expert_bounds` and then src1, so it cannot fault.
  - It does not pad the NVFP4 scales. Production's NVFP4 models run on MLX, not on this path.
- **Verification for the fork:** the full compat series, with the new 903, applies to a clean b11081. The run of
  b11081 with the series and the ten test cases is pending; see "Results".
