# Metal is not affected by the MMQ tail-padding defect

Asked while reviewing [#448](https://github.com/MaxusAI/ollama/pull/448): the MMQ `MUL_MAT_ID` defect is in
`ggml-cuda`, which compiles for NVIDIA, AMD and MUSA. If the bug is about MMQ rather than about CUDA, does Metal
need testing too?

**No, and there is nothing to ask the Metal host.** Metal's `MUL_MAT_ID` is unaffected, and not by luck — it
bounds its tile indices where MMQ does not. Read from llama.cpp `b11081`, the fork's pin.

## The defect needs three things, and Metal has none of them

The CUDA bug ([upstream-mmq-29953-material.md](upstream-mmq-29953-material.md)) is: a buffer is allocated with a
tail padding, a tile reads a fixed number of columns from it **unconditionally**, and the padding has to be at
least as large as the read. Get the padding wrong and the read leaves the allocation.

| | `ggml-cuda` MMQ | `ggml-metal` |
|---|---|---|
| a requantized, tiled src1 buffer | `src1_q8_1`, padded by a tail term | **none**: src1 is read in place |
| a compacted expert→row map | `ids_dst`, `ne12*n_expert_used` entries, expert ranges packed back to back | `hids`, `ne02*ne21` -- **one slot per (expert, token), uncompacted** |
| unconditional whole-tile loads | yes, `for (l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += nthreads) tile_y[l] = by0[l];` | **no, the index is clamped** |

### Metal allocates no src1 copy

`ggml_backend_metal_buffer_type_get_alloc_size()` appends exactly three extras to a `MUL_MAT_ID` node
(`ggml-metal.cpp`, `ggml-metal-ops.cpp`):

- `extra_tpe` = `sizeof(I32) * ne02` -- tokens per expert;
- `extra_ids` = `sizeof(I32) * ne02 * ne21` -- the expert-to-token map;
- `extra_amax` = `8 + N_MM_NPART_AMAX*sizeof(float)` -- scaling factors for the amax pass.

There is no requantized src1 among them. `kernel_mul_mm_id` reads src1 in place through `args.nb1*`, so the
buffer MMQ pads most carefully has no Metal counterpart at all.

### The expert map is padded by construction

CUDA's `ids_dst` is *compacted*: each expert's rows sit immediately after the previous expert's, which is exactly
why reading `J` entries from an expert's first row runs into the next expert and, for the last expert, off the
end. Metal's `hids` is `ne02 * ne21` — **every expert owns a full `ne21`-long row whatever it was actually
routed**, so there is no tail to overrun.

### And the tile index is clamped, with a comment saying so

`kernel_mul_mm_id`, `ggml/src/ggml-metal/kernels/mul_mm.metal`:

```c
    const int32_t neh1 = tpe_u32[im];          // rows this expert actually got

    if (r1 >= neh1) {
        return;
    }

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (    neh1 - r1 < NR1) ? (    neh1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1; // 0 .. 31

    const int id = ids_i32[im*args.ne21 + r1 + lr1];
```

`lr1` is clamped to `nr1 - 1` and `nr1` is `min(neh1 - r1, NR1)`, so `r1 + lr1 <= neh1 - 1`: the read cannot pass
the expert's own row count, let alone the allocation. Threads past the valid rows re-read the last valid row and
their results are discarded. That is the opposite design choice from MMQ, which loads the whole tile and relies
on the allocation being large enough — and it is why Metal needs no padding rule to get right.

The vector path is safer still. `kernel_mul_mv_id` indexes the **original** `ids` tensor with grid-derived
indices (`iid1 = tgpig.z/args.nei0`) and does no tiling, so every index is in range by construction.

## What this does and does not say

- **It does say** the specific defect, and its whole class — an unconditional tile load over a buffer whose
  safety depends on a separately computed padding — is absent from Metal's `MUL_MAT_ID`. Clamping is checked at
  the point of use rather than reasoned about at the point of allocation, so there is no rule to get wrong.
- **It does not say** Metal's `MUL_MAT_ID` is correct in general. This is a reading of two kernels on one pin for
  one class of bug. Nothing here was run: there is no Apple hardware on the CUDA host, and nothing needed to be,
  because the question is answered by the absence of the buffers and the presence of the clamp.
- **It does not extend to `FLASH_ATTN_EXT`.** That op allocates its own extras, one of them named
  `flash_attn_ext_extra_pad`, which is at least shaped like a padding whose size has to be right. That is a
  separate question and this file makes no claim about it.

## For the fork

Nothing to do. Compat 903 is `ggml-cuda` only and cannot apply to the Metal backend; the Metal host serves
`0.35.0` with no MMQ in its payload. No ask was sent to the Metal host, and none is needed.
