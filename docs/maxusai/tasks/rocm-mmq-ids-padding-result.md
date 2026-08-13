# Result: MMQ ids-padding under-allocation on ROCm/gfx1151

Answers [rocm-mmq-ids-padding-test.md](rocm-mmq-ids-padding-test.md).

**Headline: gfx1151 reaches the faulty branch, but the payload this host is pinned to does
not contain the fault. The AMD upgrade gate is what protects us, and it now has a second,
independent reason to hold.**

Also **retracts** `rocm-mmvq-broadcast-result.md`, deleted in this commit — see §5.

## Environment

| | |
|---|---|
| Host | Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151** (RDNA 3.5), 96 GiB VRAM, 30 GiB system RAM |
| ROCm | 7.2.1 · HIP 7.2.53211 · `amdclang++` 22.0.0git (roc-7.2.1) |
| Server | `0.32.1-dynres-296eb020` (`v0.32.1-dynres.3`), `llama-server --version` → `cb295bf59` |
| Payload in production | **b9888** |
| Payload also examined | **b10091** (the gated 0.32.5 payload) |

## 1. Does AMD reach the branch? — YES

This was the brief's "most valuable thing to determine". Answered by reading the dispatch,
not by inference.

`ggml_cuda_mul_mat_id` guards its first block with `if (ne2 <= MMVQ_MAX_BATCH_SIZE)` (8). The
live shape has **`ne2 = 2040`**, so that entire block is skipped — **including the AMD-specific
early return** the brief flagged at `ggml-cuda.cu:1891`. That escape hatch cannot fire for
these shapes on any AMD part. Control falls to:

```c
if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
    ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);   // branch=mmq
    return;
}
```

and `ggml_cuda_should_use_mmq` on this part:

- `amd_wmma_available(cc)` is `RDNA4 || RDNA3`; gfx1151 is RDNA3.5, and
  `GGML_CUDA_CC_IS_RDNA3` covers RDNA3.0 **and** RDNA3.5 (`common.cuh:89`) → **true**
- inside the RDNA3 arm, the first test is `if (n_experts >= 64) return true;` — the live shape
  has **256 experts** → **returns true**

So gfx1151 takes `branch=mmq` for exactly the live shapes, and does so *before* any
type-specific size heuristic is consulted. **HIP should be named as affected in the upstream
report.**

Note the mechanism is not AMD-specific at all: `ggml_cuda_mmq_get_J_max` reduces `ne11 = 1` to
`ret = 0` *before* its loop (`min(1,512) = 1`, then `1 - 1%8 = 0`, loop `for(; ret > 0; ...)`
never executes), and nothing in that path depends on `cc`. **Every** architecture that reaches
this branch gets zero tail padding.

## 2. But b9888 — what we actually run — does not have the fault

The faulty sizing was introduced *after* our pinned payload. The two trees, same function,
same statement:

| payload | tail-padding term |
|---|---|
| **b9888** (production) | `get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq)` |
| **b10091** (0.32.5, gated) | `ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq)` |

b9888's term is a **constant** — 128 for any part where `turing_mma_available || amd_wmma_available`,
which includes gfx1151 — and does not read `ne11` at all. `ggml_cuda_mmq_get_J_max` does not
exist anywhere in b9888.

So the brief's assumption — *"the sizing expression is present in every payload we have
checked"* — does not hold for b9888. **This host is not affected**, and not because of
anything AMD-specific: it is affected-proof only by version.

## 3. What that means for the gate

[`amd-upgrade-gate.md`](../amd-upgrade-gate.md) blocks gfx1151 past 0.32.1/b9888 for reasons
unrelated to this bug. That gate now also happens to be the only thing standing between this
host and a real out-of-bounds read: **b10091 carries the faulty expression, and §1 shows
gfx1151 would reach it.** Any future upgrade must carry the fix
([`903-fix-mmq-ids-padding.patch`](903-fix-mmq-ids-padding.patch), landed on `main` as
`0aeb4666`) or land on a payload that already has it.

## 4. Empirical work — consistent, and inconclusive exactly where predicted

**Cold production reproduction (b9888).** Container restarted cold, then a single
1920×1080 image request at `num_ctx = 40960` against `qwen3.6:35b-a3b-q4_k_m`:

```
llama_context: n_batch = 2048   n_ubatch = 2048
slot print_timing: prompt eval time = 4352.89 ms / 2056 tokens
slot release: stop processing: n_tokens = 2056, truncated = 0
[GIN] 200 | 13.763457404s | POST "/api/generate"
```

`load_duration` 9.34 s confirms a genuinely cold runner. **No crash, zero HIP errors.**
Consistent with §2 — though on its own it would prove nothing, since the brief correctly warns
a pass is not evidence of absence.

**`test-backend-ops` on a HIP build of the affected payload.** Built `test-backend-ops` from
**b10091** for `gfx1151` and injected all 17 live-shape cases from
[qwen35moe-mmq-testcases.cpp](qwen35moe-mmq-testcases.cpp) into `make_test_cases_eval()`.
Result: **every vulnerable shape passes**, including `q4_K, 256 experts, n_used=8, b=true,
m=512, n=2040, k=2048` — the live fault shape — plus the `n_used ∈ {1,2,4}` sweep and the
`q4_0`/`q8_0`/`q6_K` spread. No GPU fault.

This **confirms the brief's warning on HIP as well**: the cases are not an oracle. The
over-read lands in padding rows the kernel discards, so NMSE cannot see it. A green
`test-backend-ops` run on ROCm means nothing about this bug, and should not be quoted as
evidence either way.

Two practical notes for anyone repeating this:

- The cases must go in `make_test_cases_eval()`. Inserting after the *last*
  `test_mul_mat_id` line in the file puts them in `make_test_cases_perf()`, which
  `-o MUL_MAT_ID` compiles but never evaluates — the total stayed at exactly 790/790 and
  looked like a clean run.
- Building llama.cpp standalone from ollama's vendored tree needs
  `-I<ollama>/llama/compat` **and** `llama-ollama-compat.cpp` +
  `llama-ollama-compat-util.cpp` added to the `llama` target, or the link fails on
  `llama_ollama_compat::` symbols.

No sanitizer run: ROCm has no `compute-sanitizer` equivalent here, and device-side ASAN on
gfx1151 needs xnack. Given §2 makes the production answer definitive and §1 makes the
b10091 answer definitive, a sanitizer would only add belt-and-braces.

## 5. Retraction: the earlier mmvq result

`rocm-mmvq-broadcast-result.md` (deleted here) answered the superseded brief and is wrong
in its central claim. It reported gfx1151 as affected at `ne2 ∈ {2,4}` via
`ggml_cuda_mul_mat_vec_q`. That analysis was sound *about the code it read* but aimed at the
wrong kernel: mmvq is not where the fault is, `mmvq.cu:905-913` intercepts `has_ids &&
ncols_dst > 1` before the line it blamed, and the live fault carries `ne2 = 2040`, far above
the `MMVQ_MAX_BATCH_SIZE` gate the whole argument turned on.

It also contained an independent error worth naming, since it is easy to repeat: it treated
the over-read magnitude as growing with `ne2`. It does not. `mmq.cu`-side and `mmvq.cu`-side
alike, `nchannels_dst = ne1 = n_used`, which is 8 for every one of the live shapes;
`ne2` is `ncols_dst`. Conflating the two produced a tidy "gfx1151 escapes the big case"
story that was never true.

Deleted rather than amended: it briefs a reader toward a dead hypothesis, and the corrected
content is entirely in this document.

## Report-back summary

- **llama.cpp SHA / ROCm / gfx:** `cb295bf59` (b9888 payload) · ROCm 7.2.1 · gfx1151 (RDNA 3.5)
- **Branch for a MoE vision request:** `mmq`, established statically — `ne2 = 2040 > 8` skips
  the mmvq block *and* the AMD early return; `should_use_mmq` returns true on RDNA3 at
  `n_experts = 256 ≥ 64`. `ne11 = 1`, `ne12 = 2040`.
- **Cold reproduction:** no crash on b9888, 2056-token single prompt eval, ubatch 2048.
- **Sanitizer:** not available for this target.
- **Effect of the fix patch:** not applicable to b9888, which lacks the faulty expression.
  Required for any payload from b10091 onward.
