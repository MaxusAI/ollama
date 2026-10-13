# TASK: 903 becomes llama.cpp#29953 plus the guard, at b11351 (for the CUDA and ROCm hosts)

Handed over by the Metal host on 2026-10-11. **The patch has not been compiled.** The Metal host has no nvcc or hipcc.
The CPU padding sweep has been run, natively (item 2 below). The CUDA host and the ROCm host do the verification below, and either may take the branch over.

## Why now

The compat README's plan for 903 was:
- "From the first pin that contains llama.cpp#29953", retire 903's padding;
- "Keep the guard unless upstream has taken one".

#29953 merged upstream on 2026-10-08 as `fc9ce6b9d`. The pin, b11351 (v0.40.2), predates that, so the plan's
trigger has not fired. But two of the reasons `upstream-mmq-29953-material.md` gave against a backport no longer hold:
- **#29953 is merged.** It can no longer change.
- **b11351 carries the `prec_src1` refactor** that #29953 was written against. The b11081 version of this
  concern, "a backport would be an adaptation", is down to one context line.

Upstream did not take a guard, so the guard stays.

## What 903 is in this branch

`llama/compat/903-fix-mmq-ids-padding.patch`, cut against b11351 after 001–805 in the applier's order:

1. **llama.cpp#29953 as merged** (`fc9ce6b9d`). Seven of its eight hunks apply as they are. One is adapted: b11351
   predates #29941, so the MoE branch's allocation still reads `get_J_max(src0->type, fallback, cc, ne11)` where
   master read `ne12`. That line becomes `+ src1_q8_1_padding`, as #29953 makes it.
2. **The y-tile guard, as #458 has it** (branch `docs/mmq-guard-variant`, `ff8835b79`, with the ROCm host's
   variant). One helper, `ggml_cuda_mmq_get_nbytes_y_tile()`, sizes both the shared-memory y tile and the src1
   padding. A compile-time check over all ten config tables asserts the helper covers the y load loop's extent.
   It applies cleanly on top of (1).

The result:
- src1 is padded by the launched config's padded y tile, `nbytes_y_tile_best`, through the guard's helper;
- `ids_dst` is padded by `J_best - 1`;
- the tile is chosen once, at the launch's `prec_src1`, before allocating.

The old 903's widest-tile rule, `ggml_cuda_mmq_get_J_pad()`, is gone. All ten patches apply in order to a clean
b11351, 908 after this one.

## Asked of the CUDA host (sm_120, sm_75)

1. **The build:** build the payload at this branch. The guard's `static_assert`s must pass.
2. **The CPU sweep: done on the Metal host, natively.** The CUDA host need not run it.
   - **How:** `tasks/mmq-rules-check-native.py` builds the unmodified `mmq-rules-check.cu` with Apple clang, through
     `tasks/mmq-native-shim.py`, which copies the host-only definitions out of the tree verbatim.
   - **Equivalence:** on llama.cpp `dd266785c` its output is byte-identical to the nvcc build's
     `check-rules-wide.txt`. b11351's ten config tables are byte-identical to `dd266785c`'s.
   - **Result:** on b11351 with this series, both shipped rules are covered in all 9,431,816 MUL_MAT_ID shapes and
     203,200 dense shapes, on ten architectures (`mmq-successor-results/check-rules-native-b11351-series.txt`).
   - **What that proves, and what it does not:** both shipped rules are sufficient by construction in the sweep's
     model, so this shows the model agrees with them. It is not evidence about the code. The guard below is: it
     checks what `mmq.cu` allocates with, and it runs in the build.
   - **Optional:** the nvcc build of the sweep on b11351, for a second equivalence point.
3. **The retirement gate**, from "For the fork" in `upstream-mmq-29953-material.md`, on this branch:
   - `ids16` under `guard:src1` and `guard:ids_dst` on sm_120;
   - `dense321` under `guard:src1` on sm_75;
   - each with the padding set back to `J_best` blocks as the positive control. `mmq-variant.py`'s anchors move
     with the pin.

## Asked of the ROCm host (gfx1151)

1. **The build:** hipcc. The guard must compile in the host pass (`__HIP_DEVICE_COMPILE__`, not `__CUDA_ARCH__`).
2. **The hand-routed q2_K shapes from #449**, on this branch. These are the case where the widest-tile rule was 5
   blocks short. They must pass under a guard page, with the device reporting the head's padding, and match the CPU
   backend.

## Then

If both pass:
- the compat README's 903 entry is rewritten to describe this patch;
- the retirement register records #29953 as taken, with the guard carried;
- the branch leaves draft.

If either fails, 903 stays as `main` has it.
