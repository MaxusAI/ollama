# MLX: the Linux CUDA payload ships unloadable — no ELF RPATH, and libquadmath is not bundled

### Summary

The bundled `mlx_cuda_v13` payload is **not self-contained**. It loads only where
the host already provides CUDA, cuDNN, NCCL and libquadmath on the system search
path, and fails everywhere else — including the official Docker image, where
every safetensors model is then refused with a message that reads like a missing
feature rather than a packaging fault.

That conditional is the whole reason this survives: a developer machine with
CUDA 13 and cuDNN installed satisfies the payload's dependencies from `/usr/lib`
and `/usr/local/cuda-13.0/...`, so the missing RPATH never bites. Measured here,
same libraries and same command:

| | `libmlxc.so` | `libmlx.so` |
|---|---|---|
| host with CUDA/cuDNN/NCCL installed | 1 unresolved | **0** |
| inside `ollama/ollama` (nothing system-wide) | 1 unresolved | **12** |

MLX on CUDA does work in the field — see ollama#14046, an unrelated FLUX.2 fault
on an RTX PRO 6000 Blackwell which notes that other image-generation models run
correctly on Linux/NVIDIA. This report is **not** "MLX-CUDA is broken". It is
"the bundled payload relies on the host to complete its dependency closure, and
should not".

Two independent causes, both in CMake:

1. The payload is built with `@loader_path` as its RPATH — Mach-O syntax, taken
   literally by the ELF loader, which resolves nothing. The ELF spelling is
   `$ORIGIN`. So no library in the payload can find its siblings, and the payload
   directory is on no system search path by design. `cmake/mlx/CMakeLists.txt`
   guards the setting with `if(APPLE)`, but `x/mlxrunner/mlx/CMakeLists.txt` sets
   it **unconditionally** and is directory-scoped, so it wins for the targets it
   creates and stamps the Mach-O token onto the Linux build.
2. `MLX_INCLUDE_REGEXES` bundles `gfortran` but not `quadmath`. `libgfortran`
   links against `libquadmath`, so the dependency closure is incomplete even once
   the search path is fixed.

### Symptom

```
$ ollama pull <any safetensors model>
pulling manifest
Error: this model requires MLX support, but the MLX runtime is not available
```

The runtime *is* present:

```
/usr/lib/ollama/mlx_cuda_v13/libmlx.so     146315776 bytes
/usr/lib/ollama/mlx_cuda_v13/libmlxc.so       764792 bytes
```

`server/images.go` refuses the pull when `mlx.CheckInit()` returns an error, and
at `OLLAMA_DEBUG=1` the real reason appears:

```
ERROR failed to load MLX dynamic library: path=/usr/lib/ollama/mlx_cuda_v13/libmlxc.so
DEBUG MLX is unavailable for safetensors model pull
      error="failed to load MLX dynamic library (searched: [/usr/lib/ollama …])"
```

The path resolution is correct. The `dlopen` is what fails.

### Observed fault

```
libmlxc.so -> libmlx.so                                        not found
libmlx.so  -> libnccl.so.2                                     not found
              libcublasLt.so.13                                not found
              libcufft.so.12                                   not found
              libnvrtc.so.13                                   not found
              libcudnn.so.9, libcudnn_graph.so.9,
              libcudnn_ops.so.9, libcudnn_cnn.so.9,
              libcudnn_adv.so.9, libcudnn_heuristic.so.9,
              libcudnn_engines_precompiled.so.9,
              libcudnn_engines_runtime_compiled.so.9           not found
```

**All twelve sit in the same directory as the library that needs them.**

The payload does carry an RPATH. It carries the wrong one — the Mach-O token,
written verbatim into an ELF header, where it resolves nothing:

```
$ patchelf --print-rpath libmlxc.so     # extracted from the image
@loader_path
```

Check this from outside the container. `readelf` is **not installed in the
runtime image**, so running it in there produces empty output that reads exactly
like "no RPATH" and is really "no readelf" — a false negative this report
originally published. Copy the library out (`docker cp`) and inspect it on a host
that has binutils or patchelf.

### Affected code

`cmake/mlx/CMakeLists.txt`:

```cmake
if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()
...
set(MLX_INCLUDE_REGEXES cublas cublasLt cudart cufft nvrtc nvrtc-builtins cudnn nccl openblas gfortran)
```

`x/mlxrunner/mlx/CMakeLists.txt` sets the Mach-O spelling *unconditionally*:

```cmake
set(CMAKE_INSTALL_RPATH "@loader_path")
```

That one is directory-scoped, so it also overrides whatever the parent set for
the targets it creates — on ELF it writes a literal `@loader_path` RPATH, which
resolves nothing.

### Why nothing else recovers this on Linux

`x/mlxrunner/mlx/dynamic.go` has a mechanism that looks like it should:

```go
// prependLibraryPath prepends dir to the platform's dynamic library search
// path so the linker finds colocated libmlx before any stale copies.
// Called once after successful library load.
func prependLibraryPath(dir string) { … os.Setenv("LD_LIBRARY_PATH", dir) }
```

It cannot work here, for two reasons at once. glibc reads `LD_LIBRARY_PATH` once
at process start, so `os.Setenv` does not affect a later `dlopen`. And it runs
only *after* a successful load — the step that fails.

### Environment

- `maxusai/ollama:0.32.14-rc0-dynres` (payload b10434), Linux x86_64, CUDA 13
- Reproduces identically on stock `ollama/ollama:0.32.14-rc0`. The MLX binaries
  are byte-identical in size in both images, so this is not a fork artefact.

### Verification

Clean-room in the image, **no `LD_LIBRARY_PATH` and no bind mounts** — the
shipped payload against one carrying both fixes:

| | `libmlxc.so` | `libmlx.so` |
|---|---|---|
| as shipped | 1 unresolved | **12 unresolved** |
| RPATH `$ORIGIN` + `libquadmath.so.0` | **0** | **0** |

End to end the server log flips from

```
ERROR failed to load MLX dynamic library: path=…/libmlxc.so
```

to

```
DEBUG MLX dynamic library loaded  path=/usr/lib/ollama/mlx_cuda_v13/libmlxc.so
```

`CheckInit()` passes and a safetensors pull proceeds past the MLX guard.

### What the fix reaches, and what it does not

Verified 2026-08-16 with the corrected payload and no workarounds — no
`LD_LIBRARY_PATH`, no bind mounts — pulling and loading `qwen3.5:0.8b-mlx`
(safetensors, 1.2 GB, arch `qwen3_5`):

```
DEBUG MLX dynamic library loaded    path=/usr/lib/ollama/mlx_cuda_v13/libmlxc.so
INFO  MLX engine initialized        "MLX version"=0.32.0-213-gadf21de device=gpu
INFO  Loaded tensors from manifest  count=623
INFO  mlx runner is ready           port=39675
DEBUG finished setting up           runner.vram="1.1 GiB"
```

So the payload loads, MLX initialises on the GPU, and a model is resident on
CUDA. That is the whole of what this fix claims, and it holds.

**And it serves.** Measured on an otherwise idle host, same prompt, greedy:

| model | arch | decode | notes |
|---|---|---|---|
| `gemma4:31b-nvfp4` | gemma4 | **41.5 tok/s** | 188 tokens, correct output, 22 GiB VRAM |
| `qwen3.5:0.8b-mlx` | qwen3_5 | **83.8 tok/s** | correct output with `think:false` |
| `qwen3.6:35b-a3b-nvfp4` | qwen3_5_moe | not characterised | correct output; 24 GiB VRAM |

The MoE row carries no decode rate deliberately: one 20-token run on a busy host
is not a benchmark. Its wall-clock that day — 343 s for 20 tokens — was one-time
JIT warm-up, not per-request cost; warm, it prefills 9,000 tokens in 4 s (see
prefill below).

The gemma4 figure is corroborated by the runner's own speculation controller,
which independently reported `expected_tps="0:35.2 1:47.2"` for that load.

**A warning worth not over-reading.** Both loads log, for `qwen3_5`:

```
WARN custom GPU kernel backend disabled kernel=depthwise_conv_silu backend=cuda reason="no source"
WARN custom GPU kernel backend disabled kernel=gated_delta        backend=cuda reason="no source"
```

`getCUDA()` disables a kernel when `k.cuda.source == ""`, and of the custom set
only `gated_delta_recurrence` carries a CUDA source — the rest are Metal-only.
That much is real and read from source. But the generic fallbacks are
**adequate**, and the evidence is direct: `qwen3_5` trips two of these warnings
and is the fastest model measured here, while `qwen3_5_moe` trips **three** —
`gated_delta`, `gated_delta_states` and `depthwise_conv_silu` — and still emits
correct output. The warnings mark a missing fast path, not a missing capability,
and must not be reported as a serving blocker.

Note also that a file-level grep for these kernels under `x/models/` does **not**
predict which architectures hit them: `qwen3_5_moe` shows no direct reference and
trips the most warnings of the three. Read the load log, not the call sites.

### Prefill: the fallback is not the bottleneck

The obvious worry is that the Metal-only kernels are the *chunked prefill* path
(`gated_delta`, `gated_delta_states`), so long prompts would crawl. Measured, they
do not — and a CUDA port of them would be wasted work.

Prefill throughput, `num_predict=1`, unique UUID-prefixed prompts, every shape
warmed first, median of three:

| prompt tokens | `qwen3.5:0.8b` *(gated-delta)* | `gemma4:31b` *(no gated-delta)* | `qwen3.6:35b-a3b` *(gated-delta)* |
|---|---|---|---|
| ~160 | 4,537 tok/s | 918 | 770 |
| ~605 | 11,444 | 766 | 1,790 |
| ~2,280 | 24,312 | HTTP 500 | 2,258 |
| ~9,000 | **34,475** | **201** | 2,251 |

The two architectures that hit the fallback scale cleanly. The MoE holds a flat
~2,250 tok/s from 600 to 9,000 tokens, which is what a working linear-attention
prefill looks like. `gemma4`, which never touches these kernels, is the slow one
and *degrades* with length — 44-46 s to prefill 9,000 tokens, reproducible — the
shape of full attention, not of a missing kernel.

Two loose ends this leaves, neither a packaging matter: `gemma4` returned HTTP 500
at ~2,280 tokens while ~9,000 succeeded, and its 201 tok/s at 9k is the only
genuinely slow number in the MLX-on-CUDA picture. Both deserve their own
investigation.

**Method note, because two earlier attempts produced garbage.** A first pass with
no warm-up measured JIT compilation — 53 s for 124 tokens, then 0.08 s for 572,
non-monotonic and meaningless. A second pass warmed each shape with *the same*
prompt it then measured, so prompt-cache hits reported 422 million tok/s. Only
the third design measures prefill: warm every shape, then measure with prompts
carrying a unique prefix so no cached prefix is reused. Anyone re-running this
must keep both properties or they will measure the harness instead.

That correction was earned the hard way. An earlier reading of this same setup
recorded "loads but cannot generate — 4 tokens, 6 minutes, GPU at 0%". Both
numbers were artefacts: three multi-gigabyte model pulls were saturating the host
during the run, and `num_predict` was set low enough that a thinking model spent
its whole allowance inside the thinking block and returned an empty `response`
with `eval_count == num_predict` — the trap this repo's own preflight harness
refuses to run into. On a quiet box with a sane allowance, both models generate
correctly.

So only one layer here is a defect:

| layer | state |
|---|---|
| distribution — registry gates MLX tags on reported `GOOS` | outside this repo |
| packaging — bundled payload not self-contained | **the bug; fixed here** |
| serving — MLX CUDA runs these models | works, measured above |

### Confirmed by a release build

Everything above was first established by rewriting the shipped libraries with
`patchelf`, which proves the linkage but not that the CMake change emits it. A
full release build settles that. Built from `main` at `eb0ad436`
(`0.32.14-rc0-dynres-20-geb0ad43`, same b10434 payload and same MLX pin as
upstream v0.32.14), exit 0:

| | shipped | this build |
|---|---|---|
| `libmlxc.so` RPATH | `@loader_path` | **`$ORIGIN`** |
| `libquadmath.so.0` | absent | **bundled** |
| `libmlxc.so` unresolved | 1 | **0** |
| `libmlx.so` unresolved | 12 | **0** |

and running the image with no mounts, no patched binary and no
`LD_LIBRARY_PATH`:

```
DEBUG MLX dynamic library loaded    path=/usr/lib/ollama/mlx_cuda_v13/libmlxc.so
INFO  MLX engine initialized        device=gpu
INFO  Configured MLX memory limit from free device memory
                                    active="1.14 GiB" limit="82.50 GiB" previous="90.22 GiB"
```

first request HTTP 200. The `previous="90.22 GiB"` is the backend default on a
95.6 GiB card, and the applied 82.50 GiB is free memory — the second fault in
this report, also confirmed from a real build rather than a patched binary.

### Suggested fix

Give the ELF build the RPATH it needs, in both files:

```cmake
if(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
elseif(UNIX)
    set(CMAKE_BUILD_RPATH "$ORIGIN")
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
endif()
```

and complete the dependency closure:

```cmake
set(MLX_INCLUDE_REGEXES … openblas gfortran quadmath)
```

Landed here as `fix/mlx-cuda-payload-cannot-load`.

### Two things this report deliberately does not claim

**That every MLX model runs well on CUDA.** Three were measured, all producing correct output. The
Metal-only kernels are a real gap in the fast path and some architecture may yet
depend on one badly enough to matter; that would be a separate report, with its
own measurement, taken on an idle host.

**That the tarball is a safe alternative.** `scripts/build_linux.sh` builds
`ollama-linux-amd64-mlx.tar.zst` from the same `dist/linux_amd64/lib/ollama/mlx*`
artifacts, so it carries the same defect; on a CUDA-equipped host the host hides
it, exactly as above.

**That ollama#14322 covers this.** That change (merged 2026-02-19) makes the
loader try an rpath-based `dlopen` of `libmlxc` by name before searching
directories, which addresses how *ollama* finds libmlxc. The failure here is
libmlxc's own dependency chain — `libmlx.so` and the twelve CUDA libraries
beneath it — and is untouched by it.

**That the registry gate is related.** `qwen3.8:27b-nvfp4` is refused on Linux by
`registry.ollama.ai` with `412: this model requires macOS`, decided from the
client's reported `GOOS` in the `User-Agent` (`server/images.go`). That is a
distribution decision made before any local capability is consulted, and it is
independent of this bug. The two are easy to conflate: both present as "MLX is a
Mac thing", and neither is evidence for the other.
