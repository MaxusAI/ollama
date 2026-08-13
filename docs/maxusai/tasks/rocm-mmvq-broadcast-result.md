# Result: mmvq MoE broadcast over-read on ROCm/gfx1151

Answers [rocm-mmvq-broadcast-test.md](rocm-mmvq-broadcast-test.md), whose stated priority was
*"Determining which path AMD actually takes is the single most valuable output of this task."*

**gfx1151 is affected, but on a strict subset of the shapes — and it is the opposite subset
from the one intuition suggests.** The AMD escape hatch the brief hoped for does not apply to
these shapes at all. A second, unrelated guard partially covers the host instead.

This is a **static result**, established by reading the exact source our production binary was
built from. It is decisive for the dispatch question and needs no build. The empirical items
(§4 below) are still outstanding.

## Environment

| | |
|---|---|
| Host | Ryzen AI Max+ 395 / Radeon 8060S, **gfx1151** (Strix Halo, RDNA 3.5) |
| ROCm | 7.2.1 |
| Server | `0.32.1-dynres-296eb020` (`v0.32.1-dynres.3`), container `ollama-rocm` |
| llama.cpp | `llama-server --version` → **`cb295bf59`** — the exact payload the brief names as the first affected |
| Payload source read | `b9888` @ `/opt/github/MaxusAI/ollama/build/_deps/llama_cpp-src` — the tree this build's `llama-server` came from |
| Affected model class present | `qwen3.6:35b-a3b-q4_k_m` (family `qwen35moe`, Q4_K_M), also `gemma4:26b-a4b-it-q4_K_M` |

Cross-checked against `b10091` (the gated 0.32.5 payload): **identical** in every respect below.

## 1. ROCm does compile the faulty kernel — confirmed

`ggml/src/ggml-hip/CMakeLists.txt:63`

```cmake
file(GLOB   GGML_SOURCES_ROCM "../ggml-cuda/*.cu")
```

and the line itself is byte-identical to CUDA's, at `ggml/src/ggml-cuda/mmvq.cu:512`:

```c
channel_y  = ncols_dst == 1 && ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
```

The brief's corroboration holds here too: the dedicated MoE kernel in the same file
(`mmvq.cu:715`) applies the modulo unconditionally —
`const uint32_t channel_y = fastmodulo(channel_dst, nchannels_y);` — so the two kernels in one
file disagree about whether the clamp is conditional.

## 2. The AMD escape branch is unreachable for these shapes

This is the brief's "please check this first", and the answer is that the branch cannot fire.
From `ggml_cuda_mul_mat_id` (`ggml-cuda.cu`):

```c
if (ne2 <= MMVQ_MAX_BATCH_SIZE) {                              // 8
    if (ggml_is_quantized(src0->type)) {                       // Q4_K -> TRUE
        const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
        if (ne2 <= mmvq_mmid_max) {
            ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst);   // <-- the vulnerable kernel
            return;
        }
    } else {
        if (GGML_CUDA_CC_IS_AMD(cc)) {                         // <-- the hoped-for escape
            ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
            return;
        }
    }
}
```

The `GGML_CUDA_CC_IS_AMD` diversion sits in the **`else` of `ggml_is_quantized`**. The
vulnerable shapes are **Q4_K** — quantized — so control never reaches it. It protects AMD only
for *unquantized* `src0`, which is not this workload.

## 3. What does cover gfx1151, partially: the RDNA3 batch table

`get_mmvq_mmid_max_batch` dispatches by arch. gfx1151 → `cc = OFFSET_AMD + 0x1151`, and
`common.cuh:80` defines the bracket it lands in — the comment names this exact machine:

```c
#define GGML_CUDA_CC_RDNA3_5    (GGML_CUDA_CC_OFFSET_AMD + 0x1150) // AI 370, AI Max 395 laptops.
#define GGML_CUDA_CC_IS_RDNA3_5(cc) (cc >= GGML_CUDA_CC_RDNA3_5 && cc < GGML_CUDA_CC_RDNA4)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (GGML_CUDA_CC_IS_RDNA3_0(cc) || GGML_CUDA_CC_IS_RDNA3_5(cc))
```

so gfx1151 uses `get_mmvq_mmid_max_batch_rdna3`, where **`GGML_TYPE_Q4_K` returns 4**.

The guard is therefore `ne2 <= 4`, against the brief's live-captured `ne2 = {2, 4, 7}`:

| `ne2` | `ne2 <= 4` | dispatch | verdict |
|---|---|---|---|
| **2** | yes | `mul_mat_vec_q` | **AFFECTED** |
| **4** | yes | `mul_mat_vec_q` | **AFFECTED** |
| **7** | no | falls through to `should_use_mmq` / `mmf` | not this kernel |

**The inversion worth noting:** the over-read grows with `nchannels_dst`, so the *largest*
case — `ne2 = 7`, the one the brief measured at ~16 KB — is precisely the one RDNA3 routes
away. gfx1151 is exposed on the *small* over-reads and escapes the big one. A test that only
exercises the worst-case shape would conclude, wrongly, that ROCm is clean.

Contrast with the CUDA host: Ada Lovelace and newer return `MMVQ_MAX_BATCH_SIZE` (8)
unconditionally, so **all three** shapes take the vulnerable kernel there. The two platforms
are affected differently, and neither result transfers to the other.

## 4. Still outstanding

Not done, and each needs a decision before it runs:

- **Crash reproduction** (brief §2). Requires deliberately faulting the runner and restarting
  the container between attempts. This host serves production; not run unprompted.
- **`test-backend-ops -o MUL_MAT_ID`** (brief §4) with the 15 cases in
  [qwen35moe-mmvq-testcases.cpp](qwen35moe-mmvq-testcases.cpp). This is the decisive
  *correctness* instrument — it catches the silent-corruption case that a crash test misses —
  and it needs a HIP build of the llama.cpp test target. Given §3, the case matrix should be
  extended: the interesting rows for gfx1151 are `ne2 = 2` and `ne2 = 4`, not the `ne2 = 7`
  worst case the CUDA work centred on.
- **`MMID_DBG` instrumented build** (brief §3). Now largely redundant — it would confirm at
  runtime what the source says statically. Worth doing only as a check on this analysis, and
  it costs a full ~24 min gated rebuild.

## 5. Bearing on the fix

The candidate fix in [901-fix-mmvq-channel-y.patch](901-fix-mmvq-channel-y.patch) —

```c
channel_y = ids ? fastmodulo(channel_dst, nchannels_y) : channel_dst;
```

— is arch-independent and would correct gfx1151's `ne2 ∈ {2,4}` exposure by the same edit.
Nothing in this analysis argues against it.

Practical note for whoever applies it here: the patch hunk is anchored at line 514, which is
its position in the CUDA host's tree. On `b9888` the same line is at **512**, so it applies at
an offset of −2. `patch` absorbs that silently; `git apply` may need `--recount` or a reduced
context. Verify the result rather than trusting a clean exit — the surrounding lines are
identical in both trees, so a misapplied hunk would still look plausible.

For the upstream report: **HIP should be named as affected**, with the qualification that
RDNA3/RDNA3.5 hits a narrower shape range than Ada+ NVIDIA because of the per-arch
`get_mmvq_mmid_max_batch` table, not because of any AMD-specific correctness guard. The
`GGML_CUDA_CC_IS_AMD` branch is irrelevant to this bug and should not be cited as mitigation.

## Provenance

Every claim above is a direct read of `b9888` at
`/opt/github/MaxusAI/ollama/build/_deps/llama_cpp-src`, the tree that produced the running
`llama-server`, cross-checked against `b10091`. Line numbers are that tree's. No build was
performed, no production request was issued, and the container was not restarted.
