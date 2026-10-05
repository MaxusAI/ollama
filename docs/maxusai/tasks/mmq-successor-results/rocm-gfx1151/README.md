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

## After the review on #450

### CDNA3, RDNA4 and GCN: counted vs reachable (modelled)

The review asked whether CDNA3's counted shapes are reachable. `mmq-gfx1151-reach.cu --arch` runs the same sweep
with each device's dispatch, cut from the source. These are modelled, not measured: there is no such hardware here.

| arch | widest-tile rule: counted | reach MMQ | realisable | why |
|---|---|---|---|---|
| gfx1151 (measured) | 25,384 | 2,648 | 2,616 | RDNA3 sends q2_K to hipBLAS unless ≥ 64 experts or ≤ 128 tokens |
| RDNA4 (gfx1200) | 25,384 | 25,384 | 23,746 | RDNA4 takes MMQ for every type |
| CDNA3 (gfx942) | 595,200 | 595,200 | 595,200 | takes MMQ for every type and picks `J` from tokens; all ten types short |
| GCN (gfx906) | 297,600 | 297,600 | 297,600 | not a fork target; q5_0, q5_1, q2_K, q3_K and q5_K short |

CDNA3's 595,200 is #448's 576,000 multi-expert shapes plus 19,200 one-expert ones, so the exposure is real there.
Files: [gfx1200-reach.txt](gfx1200-reach.txt), [gfx942-reach.txt](gfx942-reach.txt), [gfx906-reach.txt](gfx906-reach.txt).

### Repeated guarded calls, and the leaked address space

[multi-call.txt](multi-call.txt) shows the MUL_MAT suite on `3070d927f`. Its hooks guard the dense branch, so it
makes 175 guarded MMQ calls in one process.

- 1305/1305 tests pass in 3 of 3 runs, with the fix and without it.
- So the reuse defect needs more than repeated calls. It reproduced where a range is freed and reserved again inside
  one computation with `hipMalloc` traffic between: the per-expert fallback, and `mmq-hip-vmm-reuse.hip`.
- Peak virtual memory was 8.6–9.4 GiB with the fix and 8.5–8.9 GiB without. The leaked reservations do not show
  above the suite's own allocations, and the HIP runtime reported nothing.

## A guard against re-tightening

#29953 touches no test, and the existing suite cannot see this class of bug. Three reasons:

- the pool hides the read;
- `test_mul_mat_id`'s uniform routing never leaves one column in a wide tile on RDNA3;
- compute-sanitizer does not exist for ROCm.

Two GPU-free checks close that, and both rest on the same fact: the read depends only on a config's `J` and
`nthreads`.

- **Per config, at run time:** `mmq-padding-invariant.cu` checks every config in all ten tables (2,597 configs,
  about 10 ms). See [padding-invariant.txt](padding-invariant.txt); the output is identical on b11081 and `3070d927f`.
  - It reproduces #448's per-shape verdicts.
  - It adds the tables #448's sweep did not cover. The widest-tile rule is also short on GCN: 28 configs, by up to 7
    blocks.
  - Every shortfall it finds is launchable within the device's shared memory.
  - It calls the table functions directly, so nvcc and hipcc evaluate it alike, with no gencode list.
- **At compile time:** `mmq-padding-guard.cu` makes one `static_assert` per (table, type). It states the requirement
  from the kernel's load-loop constants (`GGML_PAD(J*MMQ_TILE_Y_K, nthreads)` ints), independent of any padding rule.
  [padding-guard.txt](padding-guard.txt), hipcc, on b11081 and on `3070d927f` alike:

  | padding rule | build |
  |---|---|
  | padded tile of the launched config (#29953 now) | ok |
  | `J` blocks (#29953 as first posted) | fails in all 10 tables |
  | widest tile (903 before its amendment) | fails in cdna, gcn, rdna3_5, rdna4 |
  | widest padded tile (903 as amended) | ok |

  One assertion per table puts too much into one constant evaluation. That trips clang's
  `-fconstexpr-steps` limit, and the failure reads like a real shortfall, so the guard asserts per (table, type).
  nvcc is untested.

The prototype carries its own copy of each rule. To guard the code that ships, the padding must be a helper that the
allocation itself calls, and the guard must check that helper. That is the form drafted for compat 903.

## #455: the #29953 guard under hipcc

#455 asked whether `mmq-29953-y-tile-guard.patch` compiles, and fires, under hipcc as it does under nvcc. The patch
is compat 903's guard, rebuilt for llama.cpp#29953's head. [verify-29953-guard.txt](verify-29953-guard.txt) holds
the eight builds of `mmq.cu` from `mmq-29953-guard-verify-hip.sh`, with ROCm 7.2.1 for gfx1151:

| build | nvcc, sm_120, either form | hipcc, the patch as published | hipcc, `VARIANT=instantiate` |
|---|---|---|---|
| 1, 2, 6: head; with the guard; guard compiled out | ok | ok | ok |
| 3: helper shrunk to `J` blocks | 220 guard errors, 10 tables | 10 guard + 10 knock-on errors, 10 tables | 220 guard errors, 10 tables |
| 4: helper rounded to one warp | 220 guard errors, 10 tables | 10 guard + 10 knock-on errors, 10 tables | 220 guard errors, 10 tables |
| 5a: an rdna3_5 config at 192 threads (#455's mutation) | not run | ok: nothing to catch | ok: nothing to catch |
| 5b: the same config at 160 threads | 1 guard error, rdna3_5 (variant only) | 1 guard + 1 knock-on error, rdna3_5 | 1 guard error, rdna3_5 |
| 5c: a blackwell config at 192 threads (the nvcc script's build 5) | 1 guard error, blackwell | 1 guard + 1 knock-on error, blackwell | 1 guard error, blackwell |

- **The guard works under hipcc.** It fails the build in exactly the tables nvcc names, and only in the host pass.
  The host pass checks every table, whatever the target, so hipcc catches the blackwell mutation too. CMake's own
  compile command and the minimal `hipcc` command agree on every build.
- **Clang adds a knock-on error.** It reports one failing type per table, and then the outer
  `static_assert(ggml_cuda_mmq_y_tile_all_<table>(...) > 0)` fails with "static assertion expression is not an
  integral constant expression". Clang prints the same words when it hits `-fconstexpr-steps`, so a reader cannot tell
  the two apart without the notes. `VARIANT=instantiate` drops the outer `static_assert`: an explicitly instantiated
  class inherits every per-type guard. Clang then reports each failing (table, type) once, as nvcc does. nvcc gives
  the same counts with either form
  ([#455](https://github.com/MaxusAI/ollama/issues/455#issuecomment-5988160612)).
- **#455's mutation 5 cannot fire on rdna3_5.** Its first 256-thread config has `J = 64`, and 64 × 36 ints is
  exactly 12 × 192. At 192 threads, `GGML_PAD`'s bit mask is short only for `J` = 8, 56, 72 and 120, and rdna3_5 uses
  multiples of 16. At 160 threads the same config is short: 9216 B against 9600 B.
- **The guard costs 0.8 s of compile time and no code.** `mmq.cu` compiles in 2.89 s with it and 2.06 s without
  (median of five, interleaved). The object is byte-identical with the guard on and off, and with the variant.
- **The script survives the fold.** With a patch that already has the variant, `VARIANT=instantiate` applies
  nothing, and every build scores as expected.

## Files

| file | what | from |
|---|---|---|
| [hipcc-check.txt](hipcc-check.txt) | hipcc vs nvcc check outputs, by sha256 | `mmq-rules-check.cu`, `mmq-jpad-cost.cu` |
| [gfx1151-reach.txt](gfx1151-reach.txt) | the gfx1151 sweep with dispatch and routing; `--production` rows at the end | `mmq-gfx1151-reach.cu --production` |
| [gfx1151-production.txt](gfx1151-production.txt) | production MoE tensors under each form of 903 | `mmq-gfx1151-prod.cu` |
| [device.txt](device.txt) | HIP VMM support, granularity, limits; one deliberate fault | `mmq-hip-vmm-probe.hip` |
| [vmm-reuse.txt](vmm-reuse.txt) | VMM range reuse, `keep` vs `free` | `mmq-hip-vmm-reuse.hip` |
| [cases.tsv](cases.tsv), [cases.md](cases.md) | 12 cases × 4 forms × 3 modes × 3 runs, b11081 | `mmq-rocm-cases.sh`, `mmq-rocm-table.py` |
| [route.tsv](route.tsv), [route.md](route.md) | hand-routed cases × 4 forms × 3 modes × 2 runs, b11081 | `mmq-rocm-route.sh`, `mmq-route.cpp` |
| [head-cases.tsv](head-cases.tsv), [head-cases.md](head-cases.md) | the cases on llama.cpp#29953 at `3070d927f` | `mmq-rocm-cases.sh` |
| [head-route.tsv](head-route.tsv), [head-route.md](head-route.md) | the hand-routed cases on `3070d927f` | `mmq-rocm-route.sh` |
| [gfx1200-reach.txt](gfx1200-reach.txt), [gfx942-reach.txt](gfx942-reach.txt), [gfx906-reach.txt](gfx906-reach.txt) | the same sweep for RDNA4, CDNA3, GCN (modelled) | `mmq-gfx1151-reach.cu --arch` |
| [multi-call.txt](multi-call.txt) | 175 guarded calls in one process, pre-fix vs fixed header | `test-backend-ops -o MUL_MAT` |
| [padding-invariant.txt](padding-invariant.txt) | every config of all ten tables against four rules | `mmq-padding-invariant.cu` |
| [padding-guard.txt](padding-guard.txt) | the compile-time guard under four rules, two trees | `mmq-padding-guard.cu` |
| [verify-29953-guard.txt](verify-29953-guard.txt) | #29953's guard: eight builds of `mmq.cu`, as published and as a variant | `mmq-29953-guard-verify-hip.sh` |

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
# #29953's guard: eight builds of mmq.cu, GPU-free (about a minute each run)
REPEATS=5 $T/mmq-29953-guard-verify-hip.sh <llama.cpp clone with 3070d927f> cmake
$T/mmq-29953-guard-verify-hip.sh <llama.cpp clone with 3070d927f> hipcc
REPEATS=5 VARIANT=instantiate $T/mmq-29953-guard-verify-hip.sh <llama.cpp clone with 3070d927f> cmake
```

The head runs above also passed `CASES=` with `dense321` appended to the twelve.
