# Upstream submission kit — PR description and issue comments

Companion to [[upstream-mmq-ids-padding-issue]]. That document is the full investigation
write-up; this is what actually gets posted.

**Strategy: open a PR, not an issue.** The fix is verifiable by *reading one function* — no
hardware, no model, no reproduction. Everything that makes an issue hard to action here
(upstream cannot load our GGUF, `test-backend-ops` goes green, the cold-vs-warm reproduction
is fiddly) stops mattering when the reviewer only has to check an internal inconsistency.
Two prior reports of what may be this same bug (#19705, #24399) were closed "not planned"
after asking maintainers to reproduce an MoE crash on hardware they may not have.

Order: PR first, then a short comment on each of the two genuinely MMQ/`mul_mat_id` threads
pointing at it. Leave #22867 and #22032 alone — the links there are weaker and a comment
would be noise.

---

## 1. PR title

```
CUDA: size MMQ ids-path tail padding from the flattened row count, not ne11
```

## 2. PR description

> ### The inconsistency
>
> In `ggml_cuda_mul_mat_q()`, the `ids` (MUL_MAT_ID) branch allocates `src1_q8_1` as a data
> term plus an MMQ tail-padding term:
>
> ```c
> const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
>     ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
> ```
>
> The data term is sized for `ne12*n_expert_used` rows. The padding term is sized from `ne11`.
>
> Those are different quantities, and the function itself says so twelve lines below:
>
> ```c
> const int64_t ne11_flat = ne12*n_expert_used;
> ```
>
> `ne11_flat` is the row count the quantise kernel is then given. The `!ids` branch a few lines
> above passes `ne11` and is correct, because there `ne11` *is* the buffer's row count — which
> is likely how the `ids` branch inherited it.
>
> ### Why it matters in practice
>
> For MoE gate/up projections the activations are broadcast across experts, so `ne11 == 1` —
> this is exactly the `dedup_bcast` case the branch handles. And with `ne11 == 1`,
> `ggml_cuda_mmq_get_J_max()` returns **zero**:
>
> ```c
> int ret = std::min(ne11, int64_t(512));   // 1
> ret -= ret % 8;                           // 1 - 1 = 0
> for (; ret > 0; ret -= 8) { ... }         // body never runs
> return ret;                               // 0
> ```
>
> So the buffer gets no tail padding at all, while MMQ processes rows in tiles of up to
> `J_max = 512` and reads past the logical end.
>
> The non-broadcast case is under-allocated too, just less visibly: `ffn_down` passes
> `ne11 = n_expert_used = 8`, sizing padding for 8 rows instead of `ne12*n_expert_used`.
>
> ### The fix
>
> ```diff
>      const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
> -        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
> +        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12*n_expert_used) * sizeof(block_q8_1_mmq);
> ```
>
> ### Evidence
>
> Found by an out-of-bounds read faulting on a MoE vision model (`qwen35moe`, 256 experts, 8
> used) when a whole image arrived as one 2040-token ubatch. With a
> `cudaStreamSynchronize(ctx.stream())` inserted before the existing `cudaGetLastError()` so
> the error is attributed to the node that caused it rather than to a later dispatch:
>
> ```
> op=MUL_MAT_ID name=ffn_moe_gate-3
>   dst  [512, 8, 2040, 1]
>   src0 blk.3.ffn_gate_exps.weight : q4_K [2048, 512, 256, 1]
>   src1 attn_post_norm-3 (reshaped): f32  [2048,   1, 2040, 1]
>   src2 ffn_moe_topk-3             : i32  [   8, 2040,    1, 1]
> branch=mmq ne0=512 ne1=8 ne2=2040 ne02=256 ne11=1 ne12=2040 type=q4_K
> ```
>
> Verified with the patch: cold container, target request first, four consecutive runs, no
> fault, against two unfixed controls built from the same tree that still fault. The branch
> log confirms the same `mmq` path with identical shapes still executes, so this corrects the
> path rather than routing around it.
>
> Environment: RTX PRO 6000 Blackwell (sm_120), CUDA 13.0. Also reproduced on `cb295bf59` and
> `b4d6c7d8f`. Unaffected by `GGML_CUDA_DISABLE_GRAPHS=1`.
>
> ### Three things worth knowing before reviewing
>
> **`test-backend-ops` will not catch this.** The over-read lands in padding rows whose
> results the kernel discards, so the NMSE comparison is unaffected and output is
> byte-identical with and without the patch. The symptom is the crash only, not wrong results.
> `compute-sanitizer --tool memcheck` does flag it. An independent sanitizer run would be the
> most useful confirmation of this PR.
>
> **A passing run is not evidence of absence.** The overrun is a fixed size past the end, so it
> faults only when it crosses an unmapped page. Measured: the same request that crashes on a
> fresh pool passes once that pool has served a larger allocation. This is worth stating
> because it makes the bug look stochastic, and I suspect it is why several earlier reports
> were attributed to codegen.
>
> **Allocation cost is bounded and constant.** The added padding is at most
> `512 * sizeof(block_q8_1_mmq)` = 72 KB per call, independent of batch size — it does not
> scale with `ne12`. In the common small-batch case `get_J_max` returns less than 512 and the
> increase is proportionally smaller.
>
> ### Possibly the same bug
>
> These may be fixed by this change; I have not reproduced their exact configurations, so I am
> flagging rather than claiming:
>
> - **#24399** (sm_120 `mul_mat_q` out-of-range shared-memory store) — same file, same kernel
>   family, same hardware class. Attributed there to Blackwell int8-MMA codegen. Both
>   workarounds offered are also explained by this defect: `GGML_CUDA_FORCE_CUBLAS=ON` avoids
>   the MMQ path entirely, and requantising changes `y_block_size`/`ne10_padded` and hence the
>   allocation size, which changes whether the fixed-size overrun lands on a mapped page.
> - **#19705** (Qwen3-Coder-Next `ggml_mul_mat_id` assertion) — MoE, `mul_mat_id`, reproduces
>   only at `-ngl 99` and is fine on CPU, which is what a fault confined to this CUDA
>   allocation would look like.
> - **#18331** (Blackwell MUL_MAT illegal access, attributed to nvcc O3 codegen) — its
>   `-DCMAKE_CUDA_ARCHITECTURES=89` workaround changes both the generated code and the fatbin,
>   so it cannot separate a codegen fault from an allocation-layout one.

---

## 3. Comment for #24399

> This may be the same bug, from a different direction — I have opened <PR link> for an
> under-allocation in the `ids` branch of `ggml_cuda_mul_mat_q()`: the tail-padding term is
> sized from `ne11` while the buffer holds `ne12*n_expert_used` rows, and for MoE broadcast
> activations (`ne11 == 1`) `ggml_cuda_mmq_get_J_max()` returns 0, so there is no tail padding
> at all.
>
> What made me look here rather than at codegen: both workarounds in this thread are also
> explained by an allocation fault. `GGML_CUDA_FORCE_CUBLAS=ON` disables the MMQ kernels
> outright, so it avoids the code path rather than the miscompilation; and requantising changes
> `y_block_size` and `ne10_padded`, which changes the allocation size and therefore whether a
> fixed-size overrun lands on a mapped page.
>
> That last point may also explain the intermittency: I measured the same request crashing on a
> fresh pool and passing once the pool had served a larger allocation. If anyone here still has
> a reproduction, `compute-sanitizer --tool memcheck` should show it, and the patch is one line
> if you want to test directly.

## 4. Comment for #19705

> Possibly related — I have opened <PR link> for an out-of-bounds read in the `ids` branch of
> `ggml_cuda_mul_mat_q()`, where the MMQ tail padding is sized from `ne11` rather than the
> `ne12*n_expert_used` rows the buffer actually holds. For MoE gate/up the activations are
> broadcast (`ne11 == 1`) and the padding term evaluates to zero.
>
> The detail from this thread that fits: full GPU offload only, fine on CPU. No CPU path shares
> that allocation. It is also consistent with this being MoE-wide rather than model-specific,
> as you noted with Nemotron-3-nano and gpt-oss-120b.
>
> I found it on `qwen35moe` with a large image ubatch (`ne12 = 2040`), which is a bigger row
> count than a text batch usually produces — that may be why it surfaced there first.

---

## Notes for whoever posts this

- Replace `<PR link>` in both comments after the PR exists. Post the PR first.
- Keep the comments short and factual; do not argue about why either issue was closed.
- The PR body deliberately leads with the inconsistency rather than the crash, so it can be
  reviewed without hardware.
- If asked for a test: there is no clean one. `test-backend-ops` cannot see it (see above), so
  the honest answer is a sanitizer run, not a unit test.
