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

1. The RPATH block is guarded by `if(APPLE)` and sets `@loader_path`, which is
   Mach-O syntax. A Linux build gets **no RPATH at all**, so no library in the
   payload can find its siblings — and the payload directory is on no system
   search path by design.
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

```
$ readelf -d /usr/lib/ollama/mlx_cuda_v13/libmlxc.so | grep -iE 'rpath|runpath'
(nothing)
```

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

**The model still cannot be served, for an unrelated reason.** Two warnings at
load:

```
WARN custom GPU kernel backend disabled kernel=depthwise_conv_silu backend=cuda reason="no source"
WARN custom GPU kernel backend disabled kernel=gated_delta        backend=cuda reason="no source"
```

`getCUDA()` disables a kernel when `k.cuda.source == ""`. Across the custom
kernel set, only `gated_delta_recurrence` carries a CUDA source; `gated_delta`,
`gated_delta_states`, `depthwise_conv_silu` and `depthwise_conv_silu_bias` are
Metal-only and fall back to their generic implementations. `qwen3_5` is a
gated-delta-net architecture and depends on exactly those.

Measured consequence: a 4-token request returned nothing in ~6 minutes, with the
runner at **196% CPU and the GPU at 0% utilisation** while 1.1 GiB of weights sat
in VRAM. Loaded on the GPU, computing on the CPU.

The three layers are worth keeping apart, because only the middle one is a
packaging fault:

| layer | state |
|---|---|
| distribution — registry gates MLX tags on reported `GOOS` | outside this repo |
| packaging — bundled payload not self-contained | **fixed here, verified** |
| kernel coverage — MLX CUDA lacks these custom kernels | upstream work in `x/mlxrunner/mlx` |

This also explains ollama#14046 without contradiction: image-generation models
run on Linux/NVIDIA because they never touch the gated-delta kernels. What can be
served on CUDA is decided by **model class**, not by platform.

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

**That MLX then serves every model on CUDA.** It does not, and the fix does not
claim to change that. `qwen3_5` loads and then cannot generate, because the
gated-delta and depthwise-conv kernels it needs have no CUDA source — measured
above. Fixing the payload makes MLX *reachable* on CUDA; what runs once you are
there is a question of kernel coverage, and belongs to a different change.

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
