# MLX: the Linux CUDA payload ships unloadable — no ELF RPATH, and libquadmath is not bundled

### Summary

`mlx_cuda_v13` is built, installed and shipped in the Linux image, and cannot be
`dlopen`ed. Every safetensors model is therefore refused on Linux with a message
that reads like a missing feature rather than a packaging fault.

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

**That MLX then serves a model on CUDA.** This is a loading fault and the fix
addresses loading. Whether inference works on this path is untested — and that is
the likely reason the bug survived: as shipped, the path cannot be reached, so
nothing downstream of `dlopen` has ever run in this configuration.

**That the registry gate is related.** `qwen3.8:27b-nvfp4` is refused on Linux by
`registry.ollama.ai` with `412: this model requires macOS`, decided from the
client's reported `GOOS` in the `User-Agent` (`server/images.go`). That is a
distribution decision made before any local capability is consulted, and it is
independent of this bug. The two are easy to conflate: both present as "MLX is a
Mac thing", and neither is evidence for the other.
