# Fast CUDA dev loop (sm_120 host)

A one-line change to a `.cu` file cost a **90-minute** rebuild. It now costs **8.5 minutes**,
measured. This documents the loop and the four traps that make a careless version of it
silently validate the wrong binary.

Use this while iterating on a kernel. **Do not use it to cut a release** — see the last
section.

## Why the full build takes 90 minutes

It is not the change. It is the architecture matrix: nvcc emits code for 12–13 GPU
architectures across two CUDA stages (v12 and v13), which compete for the same 32 cores.
Measured on the same 518-edge stage:

| configuration | wall clock |
|---|---|
| 1 architecture (`build-arch89.log` #21) | **316 s** |
| 12 architectures, uncontended (`build-cublas.log` #20) | **2020 s** |
| 12 architectures, both CUDA stages contending | 3153–5006 s |

Linking `libggml-cuda.so` takes 3.0 s and the Go build 3.7 s. Neither is worth optimising;
the arch matrix is the whole cost.

Two further findings worth knowing:

- The Dockerfile already mounts a ccache at `/root/.ccache`, but the base image ships ccache
  3.7.7, which refuses `-x cu` — so the CUDA hit rate is **0%** and every object is cold.
- The final `docker` export writes a 2.52 GB tarball (167–235 s) which then has to be
  `docker load`ed separately. The dev loop skips both.

## The loop

### 1. Scratch Dockerfile — never edit the repo's

```bash
cp /opt/github/MaxusAI/ollama/Dockerfile /mnt/8TB_SN850X_RAID1_BTRFS/tmp/ollama-build/Dockerfile.dev
```

One line changes, at the `cmake -S llama/server --preset llama_cuda_v13_linux` invocation:

```diff
-    cmake -S llama/server --preset llama_cuda_v13_linux \
+    cmake -S llama/server --preset llama_cuda_v13_linux -DCMAKE_CUDA_ARCHITECTURES='75-virtual;120-virtual' \
```

The `-D` wins over the preset's `cacheVariables`, because `ollama_set_cache_default()` in
`llama/server/CMakeLists.txt` only fires when the variable is undefined.

### 2. Build only the payload, straight to a directory

```bash
docker buildx build --builder bigdisk -f /mnt/8TB_SN850X_RAID1_BTRFS/tmp/ollama-build/Dockerfile.dev --target publish-llama-server-cuda_v13 --output type=local,dest=/mnt/8TB_SN850X_RAID1_BTRFS/tmp/ollama-build/fast /opt/github/MaxusAI/ollama
```

`publish-llama-server-cuda_v13` is a `FROM scratch` stage holding just the dist directory, so
BuildKit walks only `base → cuda-13-deps → llama-server-cuda_v13`. The v12, cpu, vulkan, mlx
and Go stages never run — which also removes the contention that inflates the full build.

Output lands at `fast/lib/ollama/cuda_v13/libggml-cuda.so`.

Do **not** point `type=local` at the `llama-server-cuda_v13` stage itself: its rootfs is
`rocm/dev-almalinux-8` plus the CUDA 13 toolkit (~31 GB) and exporting it is slower than the
tar you were avoiding.

### 3. Swap the payload into a runnable image

```bash
printf 'FROM maxusai/ollama:<base-tag>\nCOPY cuda_v13/libggml-cuda.so /usr/lib/ollama/cuda_v13/libggml-cuda.so\n' > Dockerfile.swap
docker build -f Dockerfile.swap -t maxusai/ollama:dev-sm120 /mnt/8TB_SN850X_RAID1_BTRFS/tmp/ollama-build/fast/lib/ollama
```

Seconds. Prefer this over `docker cp` into a live container: it is tagged and reproducible,
and it cannot hit the SIGBUS you get by overwriting a `.so` a runner has mmap'd. If you do use
`docker cp`, stop the container first, and remember `OLLAMA_KEEP_ALIVE` defaults to 5 minutes —
a still-resident model keeps the **old** library mapped.

## Measured

| | full build | dev loop |
|---|---|---|
| wall clock | ~90 min | **507 s (8.5 min)** |
| `libggml-cuda.so` | 249,074,032 B | 59,347,280 B |
| output | 2.52 GB tar, then `docker load` | directory, ready to `COPY` |

Verified end to end: 26/26 layers offloaded to CUDA0, correct generation, warm request 1.18 s.

## Four traps

**Quote the semicolon.** `RUN` is shell-form. Unquoted, `sh` splits at `;`, configures one
architecture, then tries to execute `120a-virtual` and the layer dies with exit 127.

**Use `-virtual`, not `-real`.** The shipped `cuda_v13` preset is all-virtual: the release
`.so` is PTX-only and the driver JITs it. Building `120-real` substitutes CUDA 13.0's ptxas
for the driver JIT — a *different code generator*, so you would be validating a kernel that
release never runs. ggml rewrites `120-virtual` to `120a-virtual`, matching release exactly.

**Keep `75-virtual`.** This host has a second GPU (RTX 2080 Ti, cc 7.5). ollama's
`filterOverlapByLibrary` keeps the newest libdir that covers *all* GPUs. Drop sm_75 and
cuda_v13 covers one device while cuda_v12 covers two — so the loader **silently falls back to
the stock, unpatched cuda_v12 payload**. Both GPUs still appear, VRAM is unchanged, nothing
logs an error, and your change never executes. Always confirm:

```bash
docker logs <container> 2>&1 | grep -oE "libdirs=[a-z0-9_,]+"
```

It must read `libdirs=ollama,cuda_v13`.

**`llama-server --version` does not prove your patch is in.** The reported llama.cpp SHA comes
from the CPU build stage and is invariant to `llama/compat/*.patch` and to the arch list — a
hot-swapped container reports a clean pinned identity while running whatever you dropped in.
The preflight harness pins that SHA, so it cannot catch this either. To prove a patch landed,
compare `sha256sum` of the `.so` across the change, or put a marker string in the patch.

Relatedly: no build log contains the `llama/compat: applied <patch>` line, because
`FETCHCONTENT_QUIET` suppresses it. Add `-DFETCHCONTENT_QUIET=OFF` to the dev configure line
if you want positive confirmation that a compat patch applied.

## First load is slow, and that is normal

Because the build is PTX-only, the driver JIT-compiles on first load and caches the result in
`~/.nv/ComputeCache`. Measured here: 2m27s for the first request (JIT plus model load), 1.18 s
warm. Release behaves the same way; the dev loop does not introduce it. Do not mistake it for
a regression.

## Do not ship this build

The dev payload covers two architectures. Release must keep the full matrix, or the image
stops working on every other GPU — including the ROCm host's own targets, which come from a
different stage entirely. The dev loop exists to answer "does my kernel change work", and its
output should never be tagged as a release or pushed.
