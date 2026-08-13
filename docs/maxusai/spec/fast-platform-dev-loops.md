# SPEC — fast per-platform dev loops for backend debugging

Status: CUDA measured, ROCm and macOS derived from the build files (marked per claim).

Chasing a backend bug at full-build speed is what makes "reason harder" feel cheaper than
"measure" — and that is how a day gets spent on wrong hypotheses (ADR 0024,
[[qwen35moe-mmq-investigation]]). This spec is the other half of that decision: how to
rebuild one backend per platform in minutes, and — more important — **how to prove you are
running what you just built**, because every platform here has at least one silent fallback
that serves a stock payload while looking healthy.

## The shape, on every platform

1. **Copy the build definition to a scratch file.** Never edit the repo's `Dockerfile` or
   presets for a debugging run.
2. **Cut the target fan-out** — architectures, variants, or slices.
3. **Build only the one backend target**, not the default target.
4. **Export the artifact, not an image** — a directory, not a 2.5 GB tar.
5. **Swap the single library** into an existing image or tree.
6. **Prove the swap took effect.** Non-negotiable; see "Proving it" below.

## CUDA — measured

| | full build | dev loop | one arch |
|---|---|---|---|
| architectures | 12–13, ×2 CUDA stages | `75-virtual;120-virtual` | `120-virtual` |
| wall clock | ~90 min | **507 s** | **340 s** |

See [[fast-cuda-dev-loop]] for the full procedure. Two levers it documents:
restrict `CMAKE_CUDA_ARCHITECTURES` on the `cuda_v13` configure line, and build
`--target publish-llama-server-cuda_v13` with `--output type=local`.

A third lever, found later and applying to every platform: the Dockerfile builds the
**default** cmake target, so each GPU stage also compiles `llama-server`, `libllama`,
`libmtmd` and `common` — all of which the publish stage discards. Adding
`--target ggml-cuda` (or `ggml-hip`) skips the entire host-side C++ build.

**Trap — silent payload fallback.** `filterOverlapByLibrary`
(`discover/runner.go:438-486`) keeps the newest libdir covering *all* visible GPUs. Drop an
architecture and `cuda_v13` may stop covering a second GPU, so the loader falls back to the
**stock, unpatched `cuda_v12`** with no error. Measured: same image, `--gpus '"device=0"'`
gives `libdirs=ollama,cuda_v13`; `--gpus all` gives `cuda_v12`.

`OLLAMA_LLM_LIBRARY=cuda_v13` makes discovery skip every other libdir
(`discover/runner.go:100-102`), so a reduced-arch build fails **loudly** with no GPU instead
of silently running stock code. Prefer that over remembering a `--gpus` flag.

## ROCm — derived from the build files

The gfx1151 host runs the same `ggml-cuda` sources: `ggml/src/ggml-hip/CMakeLists.txt` globs
`../ggml-cuda/*.cu`, so a kernel patch such as `903-fix-mmq-ids-padding.patch` lands in the
ROCm payload with no source changes. One patch file serves both platforms.

**The fan-out is larger than CUDA's**: `llama/server/CMakePresets.json:216` lists **13**
`AMDGPU_TARGETS`. The repo already ships the escape hatch — `rocm_v7_2_user_arch`
(`CMakePresets.json:220-223`) is byte-identical to the shipping preset except it carries no
target list, and keeps the same `binaryDir` and `OLLAMA_RUNNER_DIR`:

```
--preset rocm_v7_2_user_arch -DAMDGPU_TARGETS=gfx1151
```

Also add `--target ggml-hip`; the repo declares a build preset with exactly that target
(`CMakePresets.json:321-325`) which the Dockerfile never uses.

**Never use `rocm_v7_2_user_arch` without `-DAMDGPU_TARGETS`.** With no list, ROCm enumerates
the *build host's* AMD GPUs — and a build container has none.

**Good news:** the `filterOverlapByLibrary` trap cannot fire on the ROCm image. `FLAVOR=rocm`
ships exactly one GPU libdir (`Dockerfile:278-280`, cpu + `rocm_v7_2`), so there is nothing to
fall back to. Nor will `filterUnsupportedROCmDevices` catch a mis-targeted build: its oracle
is the bundled rocBLAS Tensile `.dat` set, copied wholesale from the SDK and unaffected by
`AMDGPU_TARGETS`.

**The ROCm-specific trap is different and worse.** The `rocm` image takes `llama-server`,
`libllama`, `libmtmd` and `libggml-cpu` from the **separate `llama-server-cpu` stage**; the
ROCm stage contributes only `libggml-hip.so` (`Dockerfile:278-280`). So:

- a `ggml-cuda/*.cu` kernel patch → rebuild `publish-llama-server-rocm_v7_2`
- a compat patch touching `src/` or `tools/mtmd/` (001, 002, 004, 005) → rebuild
  `publish-llama-server-cpu` instead

Rebuild the wrong one and you get an image where **nothing changed**, with no error: the
layer cache rebuilds the ROCm stage (the `llama/compat` COPY busts it), while the stale CPU
stage is what actually serves.

## macOS / Metal / MLX — derived from the build files

Not Docker. A native CMake superbuild: `CMakeLists.txt:66` → `cmake/local.cmake`, one
`ExternalProject` per backend plus a Go target.

**There is no architecture list to cut.** Apple GPU targeting is expressed as Metal
language/OS version, and the fork encodes exactly two variants differing only in
`CMAKE_OSX_DEPLOYMENT_TARGET`: `mlx_metal_v3` (14.0) and `mlx_metal_v4` (26.2). The fan-out is
2 CPU architectures × MLX variants (`scripts/build_darwin.sh:30,54,61,67`), so:

```
cmake -B build . -DCMAKE_OSX_ARCHITECTURES=arm64 -DOLLAMA_MLX_BACKENDS=metal_v4 -DOLLAMA_LLAMA_BACKENDS=
cmake --build build --target <one ExternalProject>
```

Superbuild targets: `ollama-llama-server-local`, `ollama-mlx-metal_v3`, `ollama-mlx-metal_v4`,
`ollama-go`, `ollama-local`, `ollama-mlx-backends`.

**On macOS arm64 there is no swappable GPU backend at all.** The llama-server build is
`BUILD_SHARED_LIBS=OFF`, `GGML_BACKEND_DL=OFF`, `GGML_METAL_EMBED_LIBRARY=ON`
(`cmake/local.cmake:596-601`) — Metal kernels and the whole ggml stack are statically linked,
shaders embedded. The swap unit is the whole `llama-server` binary, not a `.so`. On the MLX
side the swap unit is `libmlx`/`libmlxc`/`mlx.metallib`, which leaves the GGUF-on-Metal path
byte-identical — choose the artifact to match the stack you are debugging, and the other
profile stays a valid control.

**Three macOS traps, all quiet:**

- **MLX variant fallback.** `tryLoadFromMLXSubdirs` globs `lib/ollama/mlx_*` and
  reverse-sorts, so `metal_v4` always wins; `isCompatibleMLXVariant` refuses v4 below
  macOS 26 and falls through to stock v3. Both the skip and the load are `slog.Debug`
  (`x/mlxrunner/mlx/dynamic.go:236-285`), so without `OLLAMA_DEBUG=1` there is no output
  distinguishing "loaded your rebuild" from "silently used stock". Pin with
  `OLLAMA_LLM_LIBRARY=mlx_metal_v3|v4`.
- **Development-tree search.** The MLX loader also searches `$CWD/build/lib/ollama`,
  `build/*/lib/ollama` and `dist/darwin`, rooted at the **current working directory**
  (`dynamic.go:87-168`). A test that passes from `~/` and fails from the repo checkout is
  this, not your patch.
- **Persistent Metal degradation.** On any of seven Metal init failure signatures
  (`llm/metal_retry.go:12-41`), discovery retries with `GGML_METAL_TENSOR_DISABLE=1` and
  **writes that onto the device for the rest of the session**
  (`discover/runner.go:488-521`), logged once at `slog.Warn`. A Metal kernel bug you
  introduce can quietly move the whole session onto a different code path. Set
  `GGML_METAL_TENSOR_DISABLE=1` explicitly to make that the deliberate baseline and compare.

**Also note:** the CUDA compat patches *are* applied to the source the macOS build compiles —
they just compile to nothing. `apply-patch.cmake:22` globs recursively and a failed apply is
`FATAL_ERROR`, so an unapplyable CUDA debug patch breaks the macOS build.

## Proving it — the part that is not optional

**`llama-server --version` cannot prove a patch landed.** Confirmed: `check_payload_pin`
(`preflight/checks.py:85-118`) compares `expectations.toml`'s `llama_cpp_build` against a SHA
parsed from that binary — which comes from the **CPU stage** and is invariant to
`llama/compat/*.patch` and to the arch list. `payload_pin` PASSing is compatible with any
GPU backend content whatsoever, including a stock unpatched one.

In order of cost:

1. **`libdirs=` in the log** — cheapest, catches the fallback trap:
   `docker logs <ctr> | grep -oE "libdirs=[a-z0-9_,]+"`.
2. **`/proc/<pid>/maps`** — the only thing that settles *which file* the loader mapped.
   Chain it with `sha256sum` on that exact path and compare to the host-built artifact.
   Requires a model to be loaded (the subprocess exits on unload).
3. **`cuobjdump -lptx`** — structural proof of a reduced-arch build: stock `cuda_v13` embeds
   PTX for 12 architectures, a fast-loop rebuild embeds 2. Use the **CUDA 13** binary
   (`/usr/local/cuda-13.0/bin/cuobjdump`); the host default is 12.1 and dies with
   "Invalid fatbin header". `-lelf` returning nothing is **normal** — the presets are all
   `*-virtual`, so the fatbin is PTX-only and JIT'd at load.
4. **String markers** — `grep -a -c TAG <so>` costs under a second on a 250 MB library.

**Caveat that has already bitten:** `903-fix-mmq-ids-padding.patch` contains `MAXUSAI` only in
a C comment and otherwise changes a size computation. Comments are not in the binary, so grep
can **never** prove 903 landed — hash comparison is the only option. When writing a debugging
patch, add a unique string literal purely as a build fingerprint.

**ABI:** ggml symbols are unversioned and the soname is a bare `libggml-base.so.0`, so a
backend rebuilt from the *same* `LLAMA_CPP_VERSION` drops in with no relinking. Nothing stops
a backend from a *different* revision loading against the image's `libggml-base.so.0` —
mismatched struct layouts would corrupt silently rather than fail. Keep the pin identical
across a swap.

## Rules

1. Scratch copies only; never edit the repo's build definitions for a debugging run.
2. Prove the artifact is live before believing any result — every platform has a silent
   fallback.
3. Reduced-target builds are for iteration only. Releases keep the full matrix, or the image
   stops working on every other GPU.
4. Instrumented patches live in `docs/maxusai/patches/`, **not** `llama/compat/` — that
   directory is globbed by `apply-patch.cmake` and anything in it compiles into every build.
5. Match the rebuild target to the patch class. On ROCm especially, a `src/`-or-`mtmd` patch
   rebuilt through the GPU stage changes nothing, silently.

## Unverified

- ROCm timings. The levers are read from presets and CMake; nothing has been built on
  gfx1151. Settle by running the ROCm loop once and recording wall clock.
- Whether the pinned MLX exposes a kernel-scoping or JIT option. MLX is `FetchContent`'d, not
  vendored, so it is not determinable from this repo.
- Windows. Different search key (`PATH`), an extra ROCm/Vulkan path-rewrite step, and
  `rocm_v7_1` rather than `v7_2`. Its CUDA presets already ship 6 architectures rather than
  12, so the arch lever is worth less there.
