# MMQ padding on gfx1151: the measurements behind MaxusAI/ollama#449

#449 asked the ROCm host to check, on AMD, the gfx1151 column of `mmq-rules-check.cu`. #448 had computed that
column with nvcc and run it on no GPU. This directory holds the answer, measured on `amd-server/rocm`: an
AMD Ryzen AI Max+ 395 with Radeon 8060S (gfx1151, RDNA 3.5), ROCm 7.2.1. The machine's profile is
[host-profile_mmq-successor-gfx1151.json](host-profile_mmq-successor-gfx1151.json) (ADR 0046). The summary reply
is [on #449](https://github.com/MaxusAI/ollama/issues/449#issuecomment-5985313619).

## What was found

1. **hipcc agrees with nvcc, byte for byte.**
   - `mmq-rules-check.cu` and `mmq-jpad-cost.cu`, built with hipcc on b11081 and on dd266785c, print the same
     files as the nvcc builds.
   - The sha256 sums are in [hipcc-check.txt](hipcc-check.txt).
2. **The gfx1151 gap is q2_K only, and most of it is unreachable.** Both points come from
   [gfx1151-reach.txt](gfx1151-reach.txt), from `mmq-gfx1151-reach.cu`, which adds the real dispatch to the same
   sweep:
   - All 25,384 shapes left short by the widest-tile 903 are q2_K. q3_K's gfx1151 table reaches `J = 128`, so its
     widest tile covers its `J = 80`.
   - RDNA3 sends q2_K to hipBLAS unless the op has at least 64 experts or at most 128 tokens. So 2,648 of the
     shapes reach MMQ, and 2,616 of those admit a routing that reads past the allocation.
   - None of the 128 one-expert-per-token shapes reaches MMQ, and none of the 45 dense ones does.
   - For main's 903 (#27044's rule) the counts are 29,432 counted, 6,696 reaching MMQ and 6,664 realisable.
3. **As shipped, nothing faults.** [cases.md](cases.md), stock mode:
   - The twelve cases × {b11081 as pinned, main's 903, widest tile, amended} × 3 runs pass, 144 of 144.
   - The shipped HIP build uses ggml's legacy pool (`GGML_HIP_NO_VMM` defaults ON). Its 5% look-ahead absorbs reads
     of this size.
   - On this device, 10 of the 12 cases launch `J = 16`, `ids64` launches `J = 32` and `orig2040` launches `J = 64`.
     RDNA3 picks `J` from rows per expert, so these cases never reach `J = 80`.
4. **The guard allocator runs on HIP.** [device.txt](device.txt):
   - HIP VMM is supported here, with a 4 KiB granularity.
   - A read one `int` past a mapping returns `hipErrorIllegalAddress` and logs one gfxhub page fault. The GPU is
     not reset.
   - [cases.md](cases.md), guard modes, 36 runs per cell: `guard:ids_dst` aborts every run under b11081 and main's
     903, and none under the widest and amended forms. `guard:src1` aborts 30 runs under b11081, and none under the
     three forms of 903.
5. **With a hand-built routing, the gap reproduces on the device.**
   - [route.md](route.md) covers q2_K at `J = 80`, `nthreads = 256`, need 85 blocks, with one row in the last expert.
     The shapes are 8 experts / 7 used / 74 tokens with `b` = 0 and `b` = 1, and 128 experts / 16 used / 600 tokens.
   - Under `guard:src1`, each shape aborts 2/2 under b11081, main's 903 and the widest tile, and passes under the
     amended form, which reports `pad=85`.
   - Uniform routing on the same shapes passes under every form of 903, because the last expert then holds 63 to 72
     rows.
   - The check's named worst cases are unreachable here. `65/8/8` routes every token to all 8 experts. `513/1/8`
     never calls MMQ.
   - Every passing run matches the CPU result (NMSE 4.2e-5 to 6.4e-5).
6. **The amended 903 pads 85 blocks, 720 B per call, not 86 blocks.** Its `J_pad` floors 12,288 / 144, and that is
   enough: the read ends 12,144 B past the data.
7. **Production is unaffected under all three forms.** [gfx1151-production.txt](gfx1151-production.txt):
   - The MoE tensors of qwen3.6:35b-a3b, qwen3-vl:30b-a3b, gemma4:26b-a4b and nemotron3:33b launch `J` from 16 to
     128 here, and every form of 903 covers them by at least 144 B.
   - The `q4k_e128_u6_n5_b1` row of route.md shows main's 903 short for a 6-of-128 K-quant MoE at 5 tokens. No model
     production serves has that shape: nemotron3's experts are q5_0/q8_0, which stay on MMVQ up to 8 tokens.
8. **llama.cpp#29953 at `3070d927f` is clean here.** [head-cases.md](head-cases.md) and
   [head-route.md](head-route.md):
   - 78 case runs, including #448's dense q2_K case (hipBLAS on gfx1151), and 48 hand-routed runs: 0 aborts and 0
     failures.
   - It pads src1 by the padded tile of `J_best`, and `ids_dst` by `J_best-1`.

## A caveat found on the way: HIP VMM ranges must not be reused

On this device, a VMM range that is freed and then reserved again misbehaves.
[vmm-reuse.txt](vmm-reuse.txt) shows it with `mmq-hip-vmm-reuse.hip`:

- copies and kernels read different data through the new mapping;
- writes land in other allocations, and one run ended in `HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION` in an unrelated
  kernel.

Keeping each reservation, while still unmapping and releasing the memory, is clean over 4 × 300 iterations.

`mmq-debug-alloc.cuh` freed its reservation after each call. So a process with several guarded MMQ calls read wrong
data. Upstream's per-expert fallback for `513/1/8` hit this. None of 5 runs under `guard:src1` was correct: two
returned NaN, two returned an NMSE of 1.2e-1 and 1.2e-2, and one died without a result. Under `exact:src1` and
stock, all runs were correct.

`mmq_dbg_free` now keeps the reservation on HIP, and the head runs above use the fixed header. Every other run
here makes one guarded allocation per buffer per process, so the fix does not touch them. The `MMQ_*` line count per
log is 1 everywhere else.

## Files

| file | what | from |
|---|---|---|
| [hipcc-check.txt](hipcc-check.txt) | hipcc vs nvcc check outputs, by sha256 | `mmq-rules-check.cu`, `mmq-jpad-cost.cu` |
| [gfx1151-reach.txt](gfx1151-reach.txt) | the gfx1151 sweep with dispatch and routing; `--production` rows at the end | `mmq-gfx1151-reach.cu` |
| [gfx1151-production.txt](gfx1151-production.txt) | production MoE tensors under each form of 903 | `mmq-gfx1151-prod.cu` |
| [device.txt](device.txt) | HIP VMM support, granularity, limits; one deliberate fault | `mmq-hip-vmm-probe.hip` |
| [vmm-reuse.txt](vmm-reuse.txt) | VMM range reuse, `keep` vs `free` | `mmq-hip-vmm-reuse.hip` |
| [cases.tsv](cases.tsv), [cases.md](cases.md) | 12 cases × 4 forms × 3 modes × 3 runs, b11081 | `mmq-rocm-cases.sh`, `mmq-rocm-table.py` |
| [route.tsv](route.tsv), [route.md](route.md) | hand-routed cases × 4 forms × 3 modes × 2 runs, b11081 | `mmq-rocm-route.sh`, `mmq-route.cpp` |
| [head-cases.tsv](head-cases.tsv), [head-cases.md](head-cases.md) | the cases on llama.cpp#29953 at `3070d927f` | `mmq-rocm-cases.sh` |
| [head-route.tsv](head-route.tsv), [head-route.md](head-route.md) | the hand-routed cases on `3070d927f` | `mmq-rocm-route.sh` |

## Reproduce

All tools are in `docs/maxusai/tasks/`. They need a llama.cpp clone, ROCm with hipcc, and the device. The guard
modes fault the GPU on purpose; on gfx1151 that kills only the test process, but run them with no other GPU work on
the device.

```bash
T=docs/maxusai/tasks                                   # from the root of this repository
# GPU-free checks, from a llama.cpp checkout at b11081 (161755f29)
$T/mmq-extract-dispatch.sh <llama.cpp> > dispatch.inc
hipcc -std=c++17 -DGGML_USE_HIP --offload-arch=gfx1151 -I<llama.cpp> -I<llama.cpp>/ggml/include \
      -I<llama.cpp>/ggml/src -I<llama.cpp>/ggml/src/ggml-cuda -I. $T/mmq-gfx1151-reach.cu -o mmq-gfx1151-reach
./mmq-gfx1151-reach --production
# the forms of 903 on b11081, then the twelve cases and the hand-routed ones (about 35 minutes)
$T/mmq-rocm-build.sh <llama.cpp at b11081> out
$T/mmq-rocm-cases.sh out && $T/mmq-rocm-route.sh out
$T/mmq-rocm-table.py out/cases/results.tsv
# llama.cpp#29953's head as fetched
$T/mmq-rocm-build.sh --head <llama.cpp at 3070d927f> out-head
VARIANTS=head REPS=2 $T/mmq-rocm-cases.sh out-head && VARIANTS=head REPS=2 $T/mmq-rocm-route.sh out-head
```

The head runs above also passed `CASES=` with `dense321` appended to the twelve.
