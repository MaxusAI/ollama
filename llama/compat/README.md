# llama.cpp compatibility layer

This directory holds a temporary in-process compatibility layer for existing
published Ollama GGUFs whose metadata or tensor layout does not yet match what
llama.cpp expects directly. The layer translates those files in memory at load
time so users do not need to re-pull or re-create models during the transition
to llama-server.

This patch model is intended to be short lived. The target end state is that
published models and newly created models use llama.cpp-compatible metadata and
tensor layouts on disk, and this directory can be removed.

The layer is applied automatically at build time via CMake `FetchContent`'s
`PATCH_COMMAND` for normal fetched builds. If CMake is pointed at a source
override through `FETCHCONTENT_SOURCE_DIR_LLAMA_CPP`, the same patch is applied
during configure. If `OLLAMA_LLAMA_CPP_SOURCE` is set, the patch is
intentionally skipped so a developer can iterate on a local llama.cpp tree.

## Files

- `llama-ollama-compat.h`, `llama-ollama-compat.cpp` - the compatibility
  entry points and per-architecture handlers.
- `llama-ollama-compat-util.h`, `llama-ollama-compat-util.cpp` - helpers for
  KV edits, tensor renames, skip-prefix tracking, tensor load operations, and
  small tensor repacking primitives.
- `001-llama-cpp-hooks.patch` - small additive call-site edits in llama.cpp files.
  It currently touches `src/llama-model-loader.cpp` and `tools/mtmd/clip.cpp`.
- `002-llama-cpp-nemotron-dynres.patch` - switches the nemotron_v2_vl projector
  from a fixed 512x512 canvas (256 tokens/image) to native-aspect dynamic
  resolution (256..3328 tokens, `<img>`/`</img>` markers, interpolated position
  embeddings). Fork-carried port of llama.cpp PR #23638; see
  `docs/maxusai/nemotron-dynres-patch.md`.
- `004-llama-cpp-gemma4-budget-fill.patch` - gemma4/gemma4u reference sizing:
  scale every image (up or down) so the 48-aligned patch grid fills the token
  budget, snapped down to Gemma 4's supported ladder (70/140/280/560/1120), and
  resize directly (`PAD_NONE`) instead of letterboxing. Off-ladder grids
  measurably break `box_2d` vertical grounding; see
  `docs/maxusai/gemma4-bbox-investigation-findings.md`. Makes
  `--image-min-tokens` a no-op for gemma4.

  **Upstream converging on 70/1120 is not redundancy (checked 2026-09-18, at
  b10864).** llama.cpp moved its own gemma4 defaults from
  `set_limit_image_tokens(40, 280)` to `(70, 1120)` at b10864 — exactly the
  endpoints of the ladder above — which invites the question of whether this
  patch can now go. It cannot, and the reason is that the two changes are
  orthogonal. Those bounds are inputs to `calc_size_preserved_ratio`, which
  rounds each axis to align and then clamps DOWN when over `max_pixels` or
  scales UP when under `min_pixels`; it neither snaps to the ladder nor fills
  to the budget, so an under-budget image keeps its natural rounded grid. 004
  replaces that call for gemma4 with `calc_size_budget_fill`. Upstream changed
  the bounds fed to an algorithm; this patch changes the algorithm. Two further
  checks: 004 never touches `set_limit_image_tokens` at all — that line appears
  only as patch context — and the fork passes `--image-min-tokens 70
  --image-max-tokens 1120` explicitly on every gemma4 launch, so the payload's
  own defaults never bound anything here in the first place. Re-open this
  question only if upstream adds ladder snapping or budget filling to the
  dyn_size preprocessor itself, not merely because the numbers match.

  Two related facts, so the same ground is not re-walked. The patch context
  *was* re-cut for this bump (`fb18f5c94`, "re-cut 004 for llama.cpp b10864"),
  which is why it still applies: at `2b95b4a5` its context read
  `set_limit_image_tokens(40, 280)` and on `main` it reads `(70, 1120)`.
  Separately, the gemma4 branch's `image_resize_algo = RESIZE_ALGO_BICUBIC` is
  NOT a b10864 change — b10630 already set it, so it predates the 0.33.2
  baseline. A diff of the gemma4 branch spanning b9888..b10864 shows the
  resize-algo and token-limit changes together and invites treating both as
  new; only the token limits are.

  **The MLX path does not use this patch.** The `mlx-metal` preflight profiles
  carry `patchset = []` because the compat patches do not apply to MLX at all.
  Apple Silicon gets the same geometry from `llm.BudgetFillSize`
  (`llm/llama_server.go`), a Go mirror of `calc_size_budget_fill` called from
  `mlxrunner/model/gemma4/vision.go`. The two must stay in lockstep: changing one and
  not the other silently splits GGUF and MLX onto different grids, and only the
  GGUF half is covered by this patch's tests.
- `005-llama-cpp-dynres-pinned-overshoot.patch` - shared dyn_size sizing: when
  a pinned budget (min ~= max) makes the min_pixels ceil overshoot max_pixels,
  floor just under min instead. The budget is a hard ceiling; measured
  overshoots: nemotron pinned 3328 delivered 3388, gemma4 pinned 1120
  delivered 1170 (pre-004). Affects only the infeasible pinned case.
- `801-clip-node-stats-meter.patch` - **not a compatibility shim; a diagnostic**
  (see "Number bands" below). Adds an env-gated per-node meter to the clip
  graph: with `OLLAMA_CLIP_NODE_STATS=<name substring>` (or `*`) every matching
  node's output is scanned after evaluation and its max |value|, counts above
  32k/49k/60k, and non-finite counts are logged. **Inert unless the variable is
  set** - the eval callback is not even registered otherwise, so a normal build
  and a normal run are unchanged.

- `802-lm-node-stats-meter.patch` - **the language-model twin of 801**, same
  band, same fields, so one differ reads both captures. Gated on
  `OLLAMA_LM_NODE_STATS=<name substring>` (or `*`) and **inert unless set**.

  It exists because of the gap this very README names below, under "Diagnostics
  are only useful on both sides of a comparison": localising a change to the
  encoder *or* the language model needs a meter on both, and there was one. 801
  can say "the vision tower is identical" - it did, byte-for-byte across 1346
  nodes on ROCm 7.2.4 vs 10.0.0, while 7 of 135 scored blocks still differed -
  and then triage stopped, because nothing metered where the difference actually
  was.

  **Bounded by default, unlike 801.** An LM graph carries thousands of nodes and
  is rebuilt every decode step, so an unbounded meter would write gigabytes and
  perturb the timings it is meant to explain. Only the first
  `OLLAMA_LM_NODE_STATS_GRAPHS` graphs are metered (default **1**, which is
  prefill). Every metered node is copied to host memory: this is a diagnostic
  switch, never something to leave set on a serving instance.

  It exists because the qwen2.5-vl fp16-accumulate fault
  (`docs/maxusai/qwen25vl-3b-poison-image-garbage-decode.md`, upstream
  ollama/ollama#18070) is invisible from the product surface: the observable is
  garbled text, the cause is a handful of elements out of millions overflowing
  at `v.blk.31.ffn_down`. Diagnosing it without an instrument degenerates into
  sampling, which cannot distinguish "no fault" from "not found yet". Run it
  under `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` to read magnitudes without
  overflowing; `max_abs / 65504` is the fp16 headroom. Calibrate against a known
  trigger before trusting any run - an uncalibrated meter fails silently rather
  than loudly. Usage and the trigger corpus:
  `docs/maxusai/vision-suite/synthetic-triggers/README.md`.

- `903-fix-mmq-ids-padding.patch` - **not a compatibility shim** (see "Number
  bands" below). Sizes the MMQ ids-path tail padding in
  `ggml/src/ggml-cuda/mmq.cu` from the flattened row count
  (`ne12*n_expert_used`) instead of `ne11`. Under MoE broadcast `ne11 == 1`, so
  `ggml_cuda_mmq_get_J_max()` returns 0, `src1_q8_1` gets no tail padding at
  all, and MMQ overruns the logical end by up to a 512-row tile — an illegal
  memory access. Stock-ggml defect, not fork-specific; it surfaces under Ollama
  because one value is passed as both `-b` and `-ub`, so a whole image arrives
  in a single ubatch. See `docs/maxusai/qwen35moe-mmq-investigation.md` for the
  diagnosis, `docs/maxusai/mmq-padding-regression-window.md` for the affected
  build range (b9990 is the last clean build, b9992 the first defective one, so
  this must not be backported to a lineage pinned at or below b9990 — there is
  nothing there to fix and the patch will not apply), and
  `docs/maxusai/upstream-mmq-ids-padding-issue.md` for the upstream report.
- `908-revert-fattn-mma-gemma4-tiling.patch` - **not a compatibility shim**
  (see "Number bands" below). Reverts the device half of llama.cpp
  `ce8caa6e6` ("CUDA: tune FA for Gemma 4 on Ampere or newer", in b11081):
  the MMA configuration table entries and `mma_tile_sizes` for head
  dimensions 256 and 512 in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, back to
  b10969's. With the retuned tiling, gemma4:26b-a4b think-on on CUDA left 6 of
  27 suite cases in loops that never converge at the 131072 rung; with this
  revert it leaves 1, the same count gfx1151 shows on the same b11081, where
  RDNA's own table means the tuning never applies. The same half accounts for
  every gemma4 GGUF think-off cell b11081 moved on CUDA and for one OCRBench q4
  item; reverted, both return to production's b10969 values exactly. The
  commit's host half (the Ada decode-kernel selection in `fattn.cu`) is **not**
  reverted: it moves qwen3.6's think-off cells in both directions and causes no
  loop. Measurements and the one-commit and split-half attribution:
  `docs/maxusai/tasks/upstream-sync-0.34.4.md`, "Gates 4–6 on CUDA". Measured
  on CUDA only; the file also compiles for HIP, and the check that 908 leaves
  gfx1151's kernels unchanged is the ROCm host's. Drop it when upstream retunes
  these configs without the gemma4:26b loops, as measured by the loop-rate gate
  in `docs/maxusai/retirement-register.md`.
- `compat.cmake` - CMake glue, included by `llama/server/CMakeLists.txt` (and
  `cmake/local.cmake`) before `FetchContent_Declare(llama_cpp ...)`. It sets
  `OLLAMA_LLAMA_CPP_COMPAT_PATCH_COMMAND`, which FetchContent runs as its
  `PATCH_COMMAND`: the shared idempotent applier `cmake/apply-git-patches.cmake`
  with `PATCH_DIR` pointing at this directory. The applier globs `*.patch`
  here **recursively** (so `models/` too), applies them in sorted filename
  order — the numeric prefix is the apply order — skips any patch that already
  applies in reverse (`git apply --reverse --check`, which is what makes
  re-configuring and rebuilding safe), and fails the configure (`FATAL_ERROR`)
  on the first patch that does not apply. `compat.cmake` also exports
  `OLLAMA_LLAMA_CPP_COMPAT_DIR` and `OLLAMA_LLAMA_CPP_COMPAT_SOURCES` so the
  main CMakeLists can `target_sources()` the four compat source files onto the
  fetched llama.cpp targets after `FetchContent_MakeAvailable` — the sources
  are never copied into the fetched tree.
- `models/` - the sibling **new-architecture** layer: implementations of
  architectures llama.cpp doesn't support yet, each added via a small
  registration patch. (Those files *add* archs; the files above *translate*
  existing GGUFs onto archs llama.cpp already has.)

The compatibility source files stay in this directory and are linked into the
fetched llama.cpp targets. The patch file only adds call sites.

### Number bands

`cmake/apply-git-patches.cmake` globs `*.patch` recursively and applies them
in sorted filename order, so the numeric prefix is the apply order. Three bands share
that sequence and they mean different things:

- **0xx — the compatibility layer.** Translating existing published Ollama
  GGUFs onto what llama.cpp already expects, plus the `models/` patches that
  register architectures llama.cpp does not have yet. These leave when the
  published models do.
- **8xx — diagnostics and instrumentation.** Env-gated observation code that
  is inert in a normal build and run: nothing is registered and no behaviour
  changes unless the operator sets the variable. They exist because some faults
  are not observable from the product surface, and they stay as long as the
  subsystem is worth being able to inspect. Unlike 0xx and 9xx they are not
  waiting on anything, so they are **not** expected to leave.
- **9xx — fork-carried fixes for defects in stock llama.cpp/ggml.** Not
  compatibility work: upstream-reportable bugs we are carrying until upstream
  takes the fix. They sort last so they apply on top of the compat layer, and
  they leave when the pinned `LLAMA_CPP_VERSION` moves past the fix.

The distinction matters when a patch fails to apply after a llama.cpp bump. A
0xx failure means the insertion point moved and the patch needs regenerating. An
8xx failure means the same — regenerate against the new anchor; the diagnostic
is not tracking an upstream fix. A 9xx failure often means upstream fixed the
defect, and the right response is to **delete the patch**, not re-cut it —
check before regenerating.

## Load-Time Hooks

The layer runs at a small set of loader hook points:

1. Main model constructor: `translate_metadata` inspects the parsed metadata
   and mutates the in-memory `gguf_context` and `ggml_context` when a handler
   recognizes an existing published model format. It can also request mmap
   disablement when a handler needs writable backend buffers for transformed
   tensor data.
2. Main model tensor indexing: `should_skip_tensor` hides embedded projector,
   vision, audio, MTP, or other tensors that the text loader should not claim.
3. Main model tensor reads: `maybe_load_text_tensor` applies registered
   text-side load operations, such as FFN concat or dtype promotion, before
   the normal llama.cpp file read. This is wired into full model loading
   (`load_all_data`) and single-tensor reads used by tools such as
   `llama-quantize`. Since llama.cpp b10729 replaced the whole-tensor
   `load_data_for` read with slabs via `load_data_range`, the quantize-style
   slab path goes through `maybe_load_text_tensor_range`, which materializes
   the op's full output for one active tensor at a time (single-slot cache —
   evicted when the next tensor's first range arrives) and serves each
   requested (offset, size) range from it, so quantize memory stays at one
   op tensor, matching the whole-tensor `load_data_for` read it replaced.
4. `mtmd/clip` constructor: `translate_clip_metadata` rewrites a clip-facing
   view of monolithic GGUFs into the mmproj form expected by llama.cpp.
5. `mtmd/clip` tensor load loop: `maybe_load_tensor` applies clip-side load
   operations, such as F16 to F32 promotion, QKV merge, tensor repack, tensor
   split, or zero-fill.

Files that do not match a supported published-model marker are left unchanged.
Setting `OLLAMA_LLAMA_CPP_COMPAT=0` disables the hook bodies for internal
create-time validation and for models that are already known to be
llama.cpp-compatible on disk.

## Supported Transformations

This table tracks the dispatch surface. Keep it brief; the handler comments in
`llama-ollama-compat.cpp` are the source of truth for exact KV and tensor maps.

| Internal arch / marker | Text handling | Clip/mmproj handling |
|---|---|---|
| `gemma3` | Normalizes Gemma 3 metadata, tokenizer fields, and embedded vision/projector tensors. | Gemma 3 projector translation. |
| `gemma3` + embedding markers (`embeddinggemma`) | Maps to `gemma-embedding` metadata and fixes embedding dense/norm tensors. | n/a |
| `bert` + Snowflake markers (`snowflake-arctic-embed2`) | Fixes Snowflake Arctic Embed 2 tokenizer metadata. | n/a |
| `gemma3n` | Normalizes tokenizer/EOS metadata, truncates vocab-shaped tensors, and hides unused embedded vision/audio/projector tensors. | n/a |
| `gemma4` | Normalizes tokenizer metadata and hides embedded audio/vision/projector tensors from the text loader. | Gemma 4 vision/audio projector translation for GGUF blobs. |
| `gptoss` | Maps to `gpt-oss`, copies KVs, injects missing expert FFN metadata, and renames tensors. | n/a |
| `lfm2` | Renames norm tensors and fixes feed-forward metadata. | n/a |
| `olmo3` | Maps to the OLMo2-compatible loader path. | n/a |
| `mistral3` | Fixes RoPE/YaRN metadata and hides embedded vision/projector tensors. | Pixtral-style projector translation. |
| `qwen35`, `qwen35moe` | Fixes Qwen3.5/Qwen3-VL-style text metadata, translates embedded MTP tensors, and hides embedded vision/projector tensors. | Qwen3-VL merger-style projector translation. |
| `qwen3next` | Normalizes hybrid attention KV-head metadata and renames SSM dt tensors to the names expected by llama.cpp. | n/a |
| `qwen25vl` | Maps to `qwen2vl` metadata conventions. | Qwen2.5-VL projector translation. |
| `qwen3vl`, `qwen3vlmoe` | Adds missing Qwen3-VL metadata and hides embedded vision/projector tensors. | Qwen3-VL projector translation, including QKV merge and patch-embedding split/repack. |
| `deepseekocr` | Maps to `deepseek2-ocr`, injects missing OCR/MoE metadata, and hides embedded SAM/vision/projector tensors. | DeepSeek OCR projector translation. |
| `glmocr` | Maps GLM OCR metadata/tensors to the llama.cpp-compatible view. | GLM OCR projector translation. |
| `glm4moelite` | Maps GLM-4.7 Flash MLA metadata to the `deepseek2` path and fixes special-token metadata. | n/a |
| `laguna` | Renames legacy attention-gate tensors and SWA RoPE metadata to current llama.cpp names. | n/a |
| `nemotron_h_moe` | Fixes latent-FFN variants and hides MTP tensors. | n/a |
| `nemotron_h_omni` | Selects the Nemotron text loader and hides audio/vision/projector tensors from the text loader. | Nemotron V2 VL projector translation; audio remains disabled. |
| `llama` with Llama 3 markers | Fixes Llama 3 tokenizer metadata. | n/a |
| `llama4` | Hides embedded vision/projector tensors from the text loader. | Llama 4 projector translation. |
| `clip` projector without `clip.projector_type` | n/a | Defaults LLaVA/BakLLaVA projectors to `clip.projector_type=mlp`. |

Usage:

```sh
llama-server --model /path/to/ollama-blob --mmproj /path/to/ollama-blob
```

Passing the same monolithic GGUF as both `--model` and `--mmproj` works because
each loader applies its own translation.

Additional architectures are added by implementing a `handle_<arch>()` and,
for vision models, `handle_<arch>_clip()` in `llama-ollama-compat.cpp`, then
dispatching them from `translate_metadata` / `translate_clip_metadata`. For
monolithic vision models, also update the `compatClipArches` allowlist in
`llm/llama_server.go` so Ollama passes the main GGUF as `--mmproj`.

## Regenerating the Patch Files

After a llama.cpp bump moves the insertion points, re-apply the edits to a
fresh checkout and re-cut each patch from its own commit. Each patch owns a
fixed set of files:

| Patch | Files |
|---|---|
| `001-llama-cpp-hooks.patch` | `src/llama-model-loader.cpp`, `tools/mtmd/clip.cpp` |
| `002-llama-cpp-nemotron-dynres.patch` | `tools/mtmd/clip.cpp`, `tools/mtmd/models/nemotron-v2-vl.cpp`, `tools/mtmd/mtmd.cpp` |
| `models/003-llama-cpp-laguna-metal.patch` | `src/models/laguna.cpp` |
| `004-llama-cpp-gemma4-budget-fill.patch` | `tools/mtmd/clip-model.h`, `tools/mtmd/clip.cpp`, `tools/mtmd/mtmd-image.cpp` |
| `005-llama-cpp-dynres-pinned-overshoot.patch` | `tools/mtmd/mtmd-image.cpp` |
| `903-fix-mmq-ids-padding.patch` | `ggml/src/ggml-cuda/mmq.cu` |

Clone at the pinned tag **with history** — `git apply --3way` needs the
pre-image blobs, and against a `--depth 1` clone it fails with "repository
lacks the necessary blob", falls back to a direct apply, and reports the drift
as an ordinary rejection:

```sh
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout "$(cat /path/to/ollama/LLAMA_CPP_VERSION)"
```

Apply the patches in numeric order, committing after each, so every patch is
cut against the tree its predecessors produced — 004 and 005 both touch
`mtmd-image.cpp`, and cutting them from a shared working tree merges them:

```sh
git apply --3way /path/to/ollama/llama/compat/004-llama-cpp-gemma4-budget-fill.patch
# resolve any conflicts, then re-cut from the commit:
git commit -am 004
git diff HEAD~1 HEAD -- tools/mtmd/clip-model.h tools/mtmd/clip.cpp tools/mtmd/mtmd-image.cpp \
    > /path/to/ollama/llama/compat/004-llama-cpp-gemma4-budget-fill.patch
```

Then verify, in this order — the second step is not optional:

1. **They apply.** Configuring Ollama's build tree runs the applier:

   ```sh
   cmake -S llama/server -B /tmp/patch-check -DCMAKE_BUILD_TYPE=Release -DOLLAMA_RUNNER_DIR=
   ```

2. **They compile.** A clean 3-way application proves nothing about the
   surrounding code: b10353 moved `calc_size_preserved_ratio`'s parameters into
   a `calc_size_opt` struct, and 005's hunk applied without a conflict while
   still referencing the now-nonexistent bare `max_pixels` and `align_size`.
   Building `mtmd` from the tree configured above pulls in `llama` and so
   compiles every file the **0xx** patches touch, 001 included:

   ```sh
   cmake --build /tmp/patch-check --target mtmd -j8
   ```

   `mtmd` does **not** reach the 9xx band. Those patch `ggml/src/ggml-cuda/*`,
   which is compiled only when a GPU backend is enabled, and the configure above
   enables none. Compiling all of ggml-cuda to check one hunk is disproportionate
   — it is ~140 translation units per architecture — so configure with a backend
   and build just the objects for the patched sources. Under Ninja the object is
   discoverable from the source path:

   ```sh
   cmake -S llama/server -B /tmp/cuda-patch-check -G Ninja \
       -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
       -DCMAKE_CUDA_ARCHITECTURES=80-virtual -DOLLAMA_RUNNER_DIR=
   obj=$(ninja -C /tmp/cuda-patch-check -t targets all \
       | grep -oE '[^ :]*ggml/src/ggml-cuda/mmq\.cu\.o' | head -1)
   ninja -C /tmp/cuda-patch-check "$obj"
   ```

   Swap `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=…` for
   `-DGGML_HIP=ON -DCMAKE_HIP_PLATFORM=amd -DAMDGPU_TARGETS=gfx1151` to check the
   same sources under ROCm; ggml-hip compiles the ggml-cuda tree through HIP, so
   either backend exercises the hunk.

   To iterate against a standalone llama.cpp checkout instead, build 002/004/005
   only — 001 references `llama-ollama-compat.h`, which only Ollama's build tree
   supplies:

   ```sh
   cmake -S . -B build -DGGML_METAL=OFF -DLLAMA_BUILD_TESTS=OFF \
       -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
   cmake --build build --target mtmd -j8
   ```

CI runs both steps on every pull request and every push to `main`: `test.yaml`'s
`patches` job configures on Linux and Windows and builds `mtmd` on Linux, and its
`patches-ggml` job compiles the objects for any ggml sources the patch set
touches, in a CUDA container. Both are deliberately ungated.

Do not rely on `test.yaml`'s `linux` job for this. Its CUDA and ROCm presets do
build the full backends, but the job is gated on `vars.SELF_HOSTED_RUNNERS` and
is skipped outright when no self-hosted runners are attached.

Frozen release lineages pin their own `LLAMA_CPP_VERSION` and carry their own
copies of these files. A regenerated patch targets one pinned tag and must not
be forward-ported to a branch whose pin has not moved.

## Implementation Notes

The compatibility code is mostly written against public APIs (`gguf.h`,
`ggml.h`, `ggml-backend.h`). A few operations rely on implementation details
because the public API does not expose equivalent mutators:

| Dependency | Use | Replacement if needed |
|---|---|---|
| Direct writes to `ggml_tensor::type` / `ne[]` / `nb[]` | Post-creation tensor reshape/retype for in-memory translation. | Add public tensor shape/type mutators. |
| `const_cast<char *>(gguf_get_tensor_name(...))` in `rename_tensor` | Renames gguf tensors in place. | Add a public `gguf_rename_tensor` helper. |
| `llama_model_loader` forward declaration from `src/llama-model-loader.h` | Opaque key for per-loader registries. The pointer is never dereferenced. | Replace registry keys with `const void *`. |

Two helpers need extra context:

- `reclaim_slot_as` repurposes an orphaned tensor slot as a synthesized tensor
  when a clip handler splits one source tensor into multiple destination
  tensors. This is needed because clip metadata loading allocates exactly enough
  tensor slots for the source file.
- Load-op registry overrides ignore the caller-provided `file_offset` when a
  registered operation exists. The operations capture their own source offsets
  at translation time, before renames change tensor names.

## Verifying that a patch actually reached a build

A compat patch that silently fails to apply produces a build that looks correct
and measures like the unpatched one. Two things that appear to prove application
and do not:

- **The build log.** `FetchContent`'s `PATCH_COMMAND` output is suppressed
  (`FETCHCONTENT_QUIET`), so `apply-git-patches.cmake`'s
  `"llama/compat: applied <file>"` lines appear in **no** Docker build log, for
  any patch. Grepping for them returns zero on a healthy build too.
- **`/usr/lib/ollama/llama-server`.** It does not carry `clip.cpp`. Two builds
  differing only in a `clip.cpp` patch ship byte-identical `llama-server`
  binaries.

**Hash the artifact that carries the patched translation unit:**

| patched source | artifact to hash |
|---|---|
| `tools/mtmd/clip.cpp` (001, 002, 004, 005, 801, 905) | `/usr/lib/ollama/libmtmd.so*` |
| `src/llama-context.cpp` (802) | `/usr/lib/ollama/libllama.so*` — **not** `libmtmd.so`, and not `llama-server` |
| `ggml/src/ggml-cuda/*` (903, 906) | `/usr/lib/ollama/rocm_v7_2/libggml-hip.so`, or the CUDA equivalent |
| `ggml/src/ggml.c` (907) | `/usr/lib/ollama/libggml-base.so*` |

Glob the version suffix rather than hardcoding it — `libmtmd.so.0.4.0` exists on
b10864 and not on every payload, and `sha256sum` on a missing path prints
nothing, which reads exactly like "no difference".

```sh
docker run --rm --entrypoint sh "$IMAGE" -c 'sha256sum /usr/lib/ollama/libggml-base.so.*'
```

A patch that adds a string literal can also be grepped directly, which is the
cheapest positive check available:

```sh
docker run --rm --entrypoint sh "$IMAGE" -c \
  'grep -ac OLLAMA_CLIP_NODE_STATS /usr/lib/ollama/libmtmd.so.0.4.0'   # 801
```

Measured 2026-09-20: a compat 905 experiment was nearly discarded as "the patch
did not apply" on the strength of the two checks above — no patch lines in the
log, identical `llama-server`. `libmtmd.so` differed; the patch had applied and
the experiment was valid. Verifying the wrong artifact discards good results as
readily as it accepts bad ones.

## Diagnostics are only useful on both sides of a comparison

The 8xx band (`801-clip-node-stats-meter`) is carried on `main` and **was never
backported to `release/0.32.1-dynres`**. That was reasonable while the release
lineage was frozen, and it became a blocker the moment a payload bump needed
explaining: localising a 2026-09 vision change to the encoder or the language
model needs the node meter on **both** builds, and it exists on one. The
experiment reported "encoder output differs" purely because the baseline emitted
no lines at all.

Before retiring or freezing a lineage, carry the 8xx diagnostics onto it, or
archive an instrumented build alongside the shipped one. A diagnostic that only
exists on the build you are trying to explain cannot explain it.

## Build from a worktree, not the shared tree

`docker build` reads the working tree at the moment each stage runs. A `git
checkout` or `rebase` on another branch **while a build is in flight** silently
changes what that build compiles.

Measured 2026-09-20: a build script doing `git checkout main` raced a `rebase`
run in the same tree, and the image labelled "main" was built from an unrelated
docs commit. It happened not to matter — the delta was markdown and host-side
Python — but only because the commit hashes were checked afterwards. The image
tag said one thing and the artifact was another, and nothing in the build output
disagreed.

`git worktree add --detach <dir> <ref>` and build from there. The build then has
its own tree and cannot be moved under it.

## A compat A/B needs an arm the patch cannot affect

Verifying that a patch applied is not the same as verifying that the two images
differ **only** by the patch. A negative control — a model or path the patch
provably cannot reach — is what tells them apart.

Measured 2026-09-20: a clean-looking A/B of compat 907 showed no change on the
model it targeted, which would have been published as "the patch costs nothing".
The control model then moved deterministically, which is impossible if the arms
differed only by that patch. Chasing it produced the actual finding: the patch
DOES reach the control, the premise that excluded it was wrong, and on the
control the patch is harmful. The null result was correct and the conclusion
drawn from it would have been wrong.
