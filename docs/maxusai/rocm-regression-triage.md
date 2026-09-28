# Triaging a gfx1151 regression: which instrument answers which question

Written 2026-09-21, after a ROCm 7.2.4 → 10.0.0 comparison spent most of its
effort building the instruments rather than reading them. The instruments now
exist and have recorded baselines; this is the decision procedure for using them.

The premise: **"it regressed" is not a finding.** Performance and accuracy fail
for different reasons, each instrument answers exactly one question, and reading
the wrong one produces a confident wrong answer — which happened twice in the
cycle that produced this document.

## Step 0 — is it real?

Before diagnosing anything, establish the noise floor. Both are cheap and both
have been measured on this host:

| axis | floor | how it was measured |
|---|---|---|
| throughput | **±0.4%** | two direct-I/O A/B pairs — same image, same ROCm, a knob with no score effect |
| scored output | **0 of 108 blocks** | the same two pairs, think off, temperature 0 |

Anything inside those bands is the instrument, not the result. A single sample
is never enough for throughput: `scene_single` is the first request after a
restart and gave a spurious −87% prefill reading that the median over 27 blocks
turned into +12%.

**Re-take the floor on a new host.** It is not portable, and SPEC H22's
conformance row says so rather than pretending otherwise.

## Step 1 — which axis moved?

```
python3 vision-suite/summarize_tps.py --a <baseline-tag> --b <candidate-tag>
```

Reports gen and prefill separately, **split by KV-cache class**, with the paired
median of per-test ratios.

The cache split is not optional. `prefill_tps` divides the whole
`prompt_eval_count` by a `prompt_eval_duration` that a cache hit collapses, so it
is bimodal: a median over mixed blocks reports the arm's *cache-hit rate* wearing
throughput's name. Unsplit, ROCm 10.0.0 read "flat on gemma4"; split, the same
data read **−5.5%** and **−10.4%** on the two gemma4 models. See SPEC H22.

## Step 2 — accuracy moved: where in the graph?

```
# capture, per arm: OLLAMA_CLIP_NODE_STATS='*' on the server, then read the log
python3 vision-suite/summarize_clip_fingerprint.py --labels "A,B" a.raw b.raw
```

Compat 801 meters every node of the vision graph and reports both a structural
fingerprint (name, op, type) and a numerical one (`max_abs`, fp16 headroom,
counts near the fp16 cliff, inf/nan). Read the diff as:

| what moved | what it means |
|---|---|
| `op` or `type` | a **payload or patch** change, not the GPU library |
| `max_abs` drift, graph unchanged | **kernel arithmetic** changed underneath |
| `n_inf` / `n_nan` appear | **breakage** — bisect, do not tune tolerances |
| nothing | the divergence is **downstream of the vision tower** |

That last row is a real answer, not a null result. ROCm 7.2.4 vs 10.0.0 is
byte-identical across all 1346 metered nodes — the captures hash the same — while
**7 of 135 scored blocks still differ**. So that difference cannot originate in
image encoding, which removes the vision tower from the search before anyone
opens it.

Baseline: `vision-suite/bench-runs/clip-fingerprint-rocm7-vs-rocm10-2026-09-21.json`.

## Step 3 — the toolchain actually under test

A rebuild against a different ROCm reshuffles kernel selection underneath every
number above, and **nothing else notices**: the ollama version string and the
llama.cpp SHA both stay put. That is what `check_toolchain_pin` exists for, and
why `rocm` split into `rocm7`/`rocm10` — two profiles on one platform were being
resolved by dict order, silently ([ADR 0040](adr/0040-rocm-10-is-experimental-until-it-is-faster.md)).

ROCm 10 SONAMEs no longer encode the release (`librocblas.so.5.6` against 7.2.4's
`librocblas.so.5.2.70204`), so the build writes a `ROCM_VERSION` stamp and the
probe prefers it, falling back to SONAME decode.

**The release is not the whole toolchain.** From 2026-09-24 the fork builds ROCm images on
AMD's Ubuntu 24.04 images ([ADR 0042](adr/0042-rocm-images-build-on-ubuntu-rocm-images.md)):
the same 7.2.4 release, but its runtime from Ubuntu packages, linked against glibc 2.39 and
GCC 13.3. `toolchain_build = "rocm-7.2.4"` passes for that build and for the AlmaLinux-built
production image alike. When two arms both read `rocm-7.2.4` and still differ, read the
`ROCM_IMAGE` stamp beside the payload (`/usr/lib/ollama/rocm_v7_2/ROCM_IMAGE`) before blaming
the code: an image that has none predates `Dockerfile.rocm` and is AlmaLinux-built.

## What these instruments do NOT cover

- **The language-model decode path is not metered.** Compat 801 covers the clip
  graph only. The ROCm 10 finding above — identical vision tower, 7 scored blocks
  moved — localises the divergence to the LM path and then stops, because nothing
  meters it. That is the most valuable gap to close next, and it needs a patch,
  not a script.
- **rocBLAS kernel logging is not the route.** For q4_K_M weights ggml-hip runs
  its own MMQ kernels and does not send most matmuls through rocBLAS, so a
  `ROCBLAS_LAYER=2` trace comes back **empty** — measured on both arms, not
  assumed. Profiling rocBLAS selection would be looking in the wrong library.
- **One host, one GPU.** Everything here is gfx1151.

## The two mistakes this document exists to prevent

1. **Reading a cache-hit rate as throughput.** A 7x `gemma4:31b` prefill
   "speedup" across 0.34.1 → 0.34.2 was published as a compute win. Cold-prefill
   compute is 167 → 172 tok/s; what changed is 5 of 27 blocks hitting cache
   against 20 of 27. Split by cache class before believing any prefill number.
2. **Believing an exit code instead of an artifact.** A 13-architecture build
   "passed" at exit 0; the arch list was then reported by grepping the build log,
   which picked up architectures from rocBLAS's own inventory. The authority is
   the binary — `llvm-objdump --offloading` on the produced `libggml-hip.so` —
   not the log and not the status.
