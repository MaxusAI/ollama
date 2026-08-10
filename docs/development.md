# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- [CMake](https://cmake.org/download/) 3.24 or newer
- C/C++ compiler: Clang on macOS, Visual Studio 2022 C++ tools on Windows, or GCC/Clang on Linux
- [Ninja](https://github.com/ninja-build/ninja/releases) in `PATH` is recommended, especially on Windows

For pure Go iteration against an existing native payload, run Ollama from the repository root:

```shell
go run . serve
```

> [!NOTE]
> Ollama includes native code compiled with CGO.  From time to time these data structures can change and CGO can get out of sync resulting in unexpected crashes.  You can force a full build of the native code by running `go clean -cache` first. 

## Native build model

For a fresh checkout, or after changing native code, build from the repository root. On macOS arm64, this builds Metal inference. On all other platforms this builds CPU-only inference. It builds the Go binary at the repository root and installs the native runtime payload under `build/lib/ollama`.

```shell
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

To install into a standard prefix layout:

```shell
cmake --install build --prefix /path/to/install
```

On all platforms except macOS arm64, to build GPU backends select the backends explicitly:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS="cuda_v13;vulkan"
cmake --build build --parallel 8
```

Supported backend values are `cuda_v12`, `cuda_v13`, `rocm_v7_1`, `rocm_v7_2`, `vulkan`, `cuda_jetpack5`, and `cuda_jetpack6`.

Use standard CMake architecture overrides to narrow GPU builds for local hardware:

```shell
# CUDA
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v13 -DCMAKE_CUDA_ARCHITECTURES=native

# ROCm / HIP
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 -DCMAKE_HIP_ARCHITECTURES=gfx1100
```

You can tune GGML build options by setting `GGML_*` values during configure. For example, to disable CUDA flash attention kernels for local debugging:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DGGML_CUDA_FA=OFF
```

## macOS (Apple Silicon)

Additional prerequisites:

MLX Metal requires the Metal toolchain. Install [Xcode](https://developer.apple.com/xcode/) first, then:

```shell
xcodebuild -downloadComponent MetalToolchain
```

## Windows

Additional prerequisites:

- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_type=exe_network)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)

For Ninja builds, run CMake from a Developer PowerShell/Command Prompt or another shell where the Visual Studio compiler is available.

> Building for Vulkan requires VULKAN_SDK environment variable:
> 
> PowerShell
> ```powershell
> $env:VULKAN_SDK="C:\VulkanSDK\<version>"
> ```
> CMD
> ```cmd
> set VULKAN_SDK=C:\VulkanSDK\<version>
> ```

## Windows (ARM)

Windows ARM does not support additional acceleration libraries at this time.

## Linux

Additional prerequisites:

- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
    - Or install via package manager: `sudo apt install vulkan-sdk` (Ubuntu/Debian) or `sudo dnf install vulkan-sdk` (Fedora/CentOS)
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)
    - OpenBLAS/LAPACK: `sudo apt install libopenblas-dev liblapack-dev liblapacke-dev` (Ubuntu/Debian)
> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.

## MLX Engine (Optional)

The MLX engine enables running safetensor based models. On macOS arm64, MLX is enabled by default. On other platforms, MLX backends are selected with `OLLAMA_MLX_BACKENDS`.

### CUDA

Requires CUDA 13+ and [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

### Local MLX source overrides

To build against a local checkout of MLX and/or MLX-C (useful for development), set environment variables before running CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

On macOS arm64:

```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c cmake -B build .
cmake --build build --parallel 8
```

For CUDA:

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

### MLX threading

MLX keeps each device's default stream in thread-local storage and the command
encoder behind it in a `thread_local` map, and every array records the stream it
was built on. An array can therefore only be evaluated on the OS thread that built
it. A goroutine is not an OS thread, so anything driving MLX has to stay on one —
and a stream must never be cached anywhere that outlives the thread that resolved
it.

The repo has two independent MLX bindings, which meet that requirement differently:

- **`x/mlxrunner/mlx`** — call `mlx.ClaimOSThread()` once during setup. It pins the
  goroutine for life and resets the Go-side stream cache so the new owner resolves
  its own. `x/mlxrunner` and `ollama create` do this in their worker init. Go has
  no goroutine-local storage, which is why the cache has to be tied to an explicit
  claim.
- **`x/imagegen/mlx`** — no claim call. Its cached streams are `__thread` in the
  cgo preamble, so each OS thread resolves its own; callers still pin (`InitMLX`
  locks the main goroutine).

Two rules follow when writing MLX tests:

- Every test goroutine that touches MLX must be on a pinned thread. In
  `x/mlxrunner` and the model packages the `skipIfNoMLX` helpers claim for you.
- `t.Run` subtests are separate goroutines, so build MLX arrays **inside** the
  subtest. Fixtures built in the parent and evaluated in a subtest will fail with
  `There is no Stream(gpu, N) in current thread`.

Failure modes differ by binding, which matters when you are chasing one:
`x/mlxrunner/mlx` installs an error-capturing handler and panics with that
message, while `x/imagegen/mlx` has none, so a failed eval leaves an unevaluated
array and faults with a SIGSEGV inside `mlx_array_data_*` instead. Either way the
process dies and every later test is hidden, so sweep per test (`-run '^Name$'` in
a loop) rather than trusting a single package run.

See [ADR 0017](maxusai/adr/0017-mlx-work-runs-on-a-permanently-claimed-os-thread.md)
and [ADR 0018](maxusai/adr/0018-imagegen-caches-mlx-streams-per-thread.md).

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```shell
go test ./...
```

## Library detection

Ollama looks for native helper binaries and acceleration libraries in installed and local development layouts:

* `../lib/ollama` for standard installs where `ollama` is under `bin/`
* `./lib/ollama` for Windows release-style payloads and local dist output
* `.` for macOS release artifacts that colocate helpers with `ollama`
* `build/lib/ollama` and `dist/<platform>/lib/ollama` for local development builds

If the libraries are not found, Ollama will not run with any acceleration libraries.
