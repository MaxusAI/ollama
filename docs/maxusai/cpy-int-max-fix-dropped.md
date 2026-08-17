# Why `fix/ggml-cuda-cpy-int-max-contiguous` was dropped

The branch carried one unique commit, `1108f169` — *"fix(ggml-cuda): gate INT_MAX asserts on
non-contiguous tensors in ggml_cuda_cpy"*, dated 2026-05-05, authored `Local Dev <local@localhost>`
with `Co-authored-by: Cursor`. It was never reviewed and never opened as a PR. It is being deleted
rather than landed.

Three independent reasons, any one of which is sufficient. The third is the one worth remembering.

## What the commit did

Two unrelated halves.

**`ml/backend/ggml/ggml/src/ggml-cuda/cpy.cu`** — deleted the two blanket asserts at the entry to
`ggml_cuda_cpy`:

```c
GGML_ASSERT(ggml_nbytes(src0) <= INT_MAX);
GGML_ASSERT(ggml_nbytes(src1) <= INT_MAX);
```

and re-applied them, plus `GGML_ASSERT(ne <= INT_MAX)`, behind `if (!contiguous_srcs)`. The stated
motive was real: Gemma 4 worst-case graph reservation (multimodal probe at max visual tokens) could
push a single tensor past 2 GiB and abort inside `ggml_backend_sched_reserve`.

**`CMakeLists.txt`** — 18 lines prepending `ROCM_PATH` onto `CMAKE_PREFIX_PATH` inside the
`if(CMAKE_HIP_COMPILER)` block.

## 1. Upstream already fixed it, and fixed it better

`e86f3c222` — *"cuda : fix copy of large tensors (ggml_nbytes <= INT_MAX assertion) (#18433)"*,
2026-01-01 — makes the identical deletion, then goes further: it promotes the **non-contiguous**
kernels to `int64_t` as well, adds the explicit `(int64_t)` cast to the block-index arithmetic, and
replaces the blanket byte cap with a per-launcher grid-dimension assert. `b10434` carries 13 of
those `GGML_ASSERT(num_blocks …)` guards.

`main` pins `LLAMA_CPP_VERSION = b10434`. Verified against that exact tag: `ggml_cuda_cpy` begins at
line 429 and contains no `nbytes … INT_MAX` assert anywhere; `contiguous_srcs` is present at line
460. A repo-wide grep for that class of assert across `ggml/src` returns zero hits — it is gone from
ggml entirely, not just from `cpy.cu`. **The abort this commit was written to fix cannot occur at
`b10434`.**

Related upstream work: PR [#18340](https://github.com/ggml-org/llama.cpp/pull/18340), issue
[#18341](https://github.com/ggml-org/llama.cpp/issues/18341); downstream reports
[ollama#13887](https://github.com/ollama/ollama/issues/13887) and
[ollama#14836](https://github.com/ollama/ollama/issues/14836).

## 2. Landing it now would be a regression

The `!contiguous_srcs` guard re-imposes a 2 GiB / 2G-element ceiling on exactly the strided path
upstream just taught to exceed it — re-breaking the long-context cases #18433 was written for.
Upstream's surviving cap is `num_blocks <= INT_MAX`, roughly `ne <= 1.37e11` elements: about 64×
looser.

## 3. It was unsafe as written — read this part even though the commit is gone

**The gating predicate is wrong.** `contiguous_srcs` is not what separates 64-bit-clean dispatch
from `int`-typed dispatch in that file. Type pairing is.

The chain opens with `if (src0->type == src1->type && contiguous_srcs)`, which absorbs every
same-type contiguous copy. So the same-type branches below it (F32→F32, F16→F16, I32→I32, BF16→BF16)
really are reachable only when `contiguous_srcs == false`, and the commit's new asserts do cover
them. That much of its reasoning held.

But the **eleven quantized cross-type branches** never reach that first test, and — unlike every
other cross-type branch — they carry no contiguity split at all. Compare, in the fork's vendored
tree:

```c
} else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_BF16) {
    if (contiguous_srcs) { ggml_cpy_scalar_contiguous_cuda<float, nv_bfloat16>(...); }
    else                 { ggml_cpy_scalar_cuda<float, nv_bfloat16>(...); }
```

against `cpy.cu:470`:

```c
} else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_Q8_0) {
    ggml_cpy_f32_q8_0_cuda
            (src0_ddc, src1_ddc, ne, ne00, ne01, ne02, nb00, nb01, nb02, nb03, ...);
```

No branch. And the callee at `cpy.cu:227` declares `const int ne, const int ne00 … const int nb13`
while `ggml_cuda_cpy` passes `int64_t` — implicit narrowing at all eleven call sites
(`cpy.cu:470, 473, 476, 479, 482, 485, 488, 491, 494, 497, 500`). Inside `cpy_f32_q` and
`cpy_q_f32`, `x_offset` and `dst_offset` are plain `int` too.

The threshold is **exactly the 2 GiB the patch was trying to lift**. For a contiguous F32 `src0`,
`nb03 == ggml_nbytes(src0)`, so the stride truncates the moment the tensor crosses 2 GiB. The two
deleted entry asserts were the only thing preventing that. `GGML_ASSERT(ne % QK8_0 == 0)` at
`cpy.cu:232` and its siblings evaluate the already-narrowed `int`, so past 2^31 elements even the
divisibility precondition is checked against a wrong value and can spuriously pass.

Reachability is not theoretical: `ggml-cuda.cu:4721-4776` advertises all eleven quantized CPY pairs
with no contiguity or size predicate, and contiguous quantized-KV-cache view writes hit them in
normal operation.

**So the commit traded a loud, diagnosable abort for silent out-of-bounds device access on quantized
copies.** That is the reason to call it unsafe rather than merely incomplete, and it is why this
note exists — the next person to meet an `INT_MAX` abort in `cpy.cu` will be tempted by exactly this
shape of fix.

If a minimal risk-contained gate is ever wanted again, the predicate has to exclude quantized types,
not just strided ones:

```c
const bool bypass = contiguous_srcs &&
    (src0->type == src1->type ||
     (!ggml_is_quantized(src0->type) && !ggml_is_quantized(src1->type)));
```

which also keeps the gate from silently widening if a future sync adds a new cross-type branch.

### Secondary: the path the commit called safe wasn't 64-bit clean either

`cpy_scalar_contiguous` computed `const int64_t i = blockDim.x*blockIdx.x + threadIdx.x;` — all three
operands are `unsigned int`, so the arithmetic happens in 32 bits and is widened only on assignment.
It wraps at `ne >= 2^32`, and because the wrapped index is small, `if (i >= ne) return;` does not
catch it: tail elements are never written, low elements are written twice from the wrong source.
That ceiling (17.2 GiB F32, 8.6 GiB F16/BF16) sits above the 2 GiB the patch targeted, so the path
did fix the stated symptom — but it was never limit-free, and nothing asserted it. Upstream fixed
precisely this with an explicit `(int64_t)` cast. Moot now: the vendored file is gone.

## 4. It was also mechanically inapplicable

Upstream `9db4bdba` — *"runner: Remove CGO engines, use llama-server exclusively for GGML models
(#16031)"* — deleted the vendored ggml sources. `ml/backend/ggml/ggml/src/ggml-cuda/` on `main` now
holds exactly one file (a template instance), against 130 on the branch. Fork changes are carried as
`llama/compat/*.patch`, applied to the fetched tree with `git apply --whitespace=nowarn` at zero
fuzz.

A dry run against the real `b10434` file: hunk 1 fails outright — GNU `patch` diagnoses it as
*"Reversed (or previously applied) patch detected!"* — and hunk 2 lands only under GNU `patch` with
fuzz 2, which `git apply` does not offer. Any compat patch would have had to be add-only, and by
§1–3 there is nothing worth adding.

## 5. The CMakeLists half is dead

`main`'s root `CMakeLists.txt` is 66 lines and contains no HIP block, no `CMAKE_HIP_COMPILER`, and no
`ROCM_PATH` reference anywhere in the repo. That configuration moved to
`llama/server/CMakePresets.json` and `cmake/local.cmake`. The hunk targets a block that no longer
exists.

## Recovery

Nothing here is unrecoverable. The commit is `1108f169fa01a56c6945cf25041ce36687de6d74`, based on
merge-base `9ba5a049` with `main`, sitting one commit above `feat/gemma4-visual-token-budgets`
(`e90d7d95`), which is unaffected and still live.

```sh
git branch fix/ggml-cuda-cpy-int-max-contiguous 1108f169fa01a56c6945cf25041ce36687de6d74
```

## Adjacent gap, not fixed here

While confirming the above: `llama/compat/903-fix-mmq-ids-padding.patch` is undocumented in
`llama/compat/README.md`, and the CI step named *"Verify patches compile"*
(`.github/workflows/test.yaml:161-185`) configures without `GGML_CUDA`/`GGML_HIP` and builds only
`--target mtmd`. No ggml-cuda source is compiled by CI at all, so `903` is application-checked but
never compiled. Tracked separately.
