# vim: filetype=dockerfile

ARG FLAVOR=${TARGETARCH}

ARG ROCMVERSION=7.2.1
# ROCm backend toolchain. AMD stopped publishing AlmaLinux ROCm images at 7.2.4
# -- rocm/dev-almalinux-8 has no tag past 7.2.4-complete -- so under TheRock the
# ROCm stage, and ONLY the ROCm stage, builds on Ubuntu. `base` and every other
# amd64 stage stay on AlmaLinux 8 so the published tarballs keep their glibc 2.28
# floor. See docs/maxusai/rocm-714-therock-upgrade.md.
# TheRock renamed the image suffix once already (-complete -> -full), so the
# whole tag is overridable, not just the version.
ARG ROCM_BACKEND_IMAGE=rocm/dev-ubuntu-24.04
ARG ROCM_BACKEND_TAG=10.0.0-full
ARG JETPACK5VERSION=r35.4.1
ARG JETPACK6VERSION=r36.4.0
ARG CMAKEVERSION=3.31.2
ARG NINJAVERSION=1.12.1
ARG VULKANVERSION=1.4.321.1

# Default empty stages for local MLX source overrides.
# Override with: docker build --build-context local-mlx=../mlx --build-context local-mlx-c=../mlx-c
FROM scratch AS local-mlx
FROM scratch AS local-mlx-c

FROM --platform=linux/amd64 rocm/dev-almalinux-8:${ROCMVERSION}-complete AS base-amd64
RUN dnf install -y yum-utils ccache gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ gcc-toolset-13-binutils \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH

FROM --platform=linux/arm64 almalinux:8 AS base-arm64
# install epel-release for ccache
RUN yum install -y yum-utils epel-release \
    && dnf install -y clang ccache git \
    && yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo
ENV CC=clang CXX=clang++

FROM base-${TARGETARCH} AS base
# ccache: the distro package (EPEL 8) is too old to cache nvcc reliably, so install
# an upstream static build. Nothing referenced ccache before this -- the cache mounts
# at /root/.ccache existed but no COMPILER_LAUNCHER was ever set and no shim was on
# PATH, so ccache was never invoked and the mounts did nothing.
ARG CCACHEVERSION=4.10.2
RUN set -eu; \
    arch="$(uname -m)"; case "$arch" in x86_64) a=x86_64 ;; aarch64) a=aarch64 ;; *) a="" ;; esac; \
    if [ -n "$a" ]; then \
        curl -fsSL "https://github.com/ccache/ccache/releases/download/v${CCACHEVERSION}/ccache-${CCACHEVERSION}-linux-${a}.tar.xz" \
          | tar -xJ -C /tmp && install -m0755 "/tmp/ccache-${CCACHEVERSION}-linux-${a}/ccache" /usr/local/bin/ccache; \
    fi; \
    ccache --version | head -1
# Wire it in. CMake initialises CMAKE_<LANG>_COMPILER_LAUNCHER from the environment,
# so this needs no change to any preset or cmake invocation.
ENV CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    CMAKE_HIP_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=25G \
    CCACHE_SLOPPINESS=locale,time_macros,include_file_ctime,include_file_mtime
ARG CMAKEVERSION
ARG NINJAVERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1
RUN dnf install -y unzip \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux$([ "$(uname -m)" = "aarch64" ] && echo "-aarch64").zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
ENV LDFLAGS=-s

#
# GPU toolchain stages — provide compilers for llama-server GPU builds
#

FROM base AS cpu-deps
RUN dnf install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH

FROM base AS cuda-12-deps
ARG CUDA12VERSION=12.8
RUN dnf install -y cuda-toolkit-${CUDA12VERSION//./-}
ENV PATH=/usr/local/cuda-12/bin:$PATH

FROM base AS cuda-13-deps
ARG CUDA13VERSION=13.0
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-}
ENV PATH=/usr/local/cuda-13/bin:$PATH

# ROCm toolchain base -- Ubuntu, and deliberately NOT a parent of `base`. It
# mirrors base's ccache/cmake/ninja setup with apt in place of dnf; nothing else
# in the file derives from it.
FROM --platform=linux/amd64 ${ROCM_BACKEND_IMAGE}:${ROCM_BACKEND_TAG} AS rocm-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git unzip xz-utils \
    && rm -rf /var/lib/apt/lists/*
ARG CCACHEVERSION=4.10.2
RUN set -eu; \
    curl -fsSL "https://github.com/ccache/ccache/releases/download/v${CCACHEVERSION}/ccache-${CCACHEVERSION}-linux-x86_64.tar.xz" \
      | tar -xJ -C /tmp \
    && install -m0755 "/tmp/ccache-${CCACHEVERSION}-linux-x86_64/ccache" /usr/local/bin/ccache \
    && ccache --version | head -1
ENV CMAKE_C_COMPILER_LAUNCHER=ccache \
    CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    CMAKE_HIP_COMPILER_LAUNCHER=ccache \
    CCACHE_DIR=/root/.ccache \
    CCACHE_MAXSIZE=25G \
    CCACHE_SLOPPINESS=locale,time_macros,include_file_ctime,include_file_mtime
ARG CMAKEVERSION
ARG NINJAVERSION
RUN curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-x86_64.tar.gz | tar xz -C /usr/local --strip-components 1
RUN curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
ENV LDFLAGS=-s

FROM rocm-base AS rocm-deps
ENV PATH=/opt/rocm/llvm/bin:/opt/rocm/bin:$PATH

FROM base AS vulkan-deps
ARG VULKANVERSION
RUN ln -s /usr/bin/python3 /usr/bin/python \
    && wget https://sdk.lunarg.com/sdk/download/${VULKANVERSION}/linux/vulkansdk-linux-x86_64-${VULKANVERSION}.tar.xz -O /tmp/vulkansdk.tar.xz \
    && tar xvf /tmp/vulkansdk.tar.xz -C /tmp \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 vulkan-headers \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 spirv-headers \
    && /tmp/${VULKANVERSION}/vulkansdk -j 8 shaderc \
    && cp -r /tmp/${VULKANVERSION}/x86_64/include/* /usr/local/include/ \
    && cp -r /tmp/${VULKANVERSION}/x86_64/lib/* /usr/local/lib \
    && cp -r /tmp/${VULKANVERSION}/x86_64/share/* /usr/local/share/ \
    && cp -r /tmp/${VULKANVERSION}/x86_64/bin/* /usr/local/bin/ \
    && rm -rf /tmp/${VULKANVERSION} /tmp/vulkansdk.tar.xz
ENV VULKAN_SDK=/usr/local

#
# llama-server stages — rebuild when LLAMA_CPP_VERSION, llama/server/, llama/compat/, or cmake/ changes.
#
# CPU stage: llama-server + ggml-base + ggml-cpu variants → lib/ollama/
# GPU stages: GPU backend .so only → lib/ollama/<variant>/
#

FROM cpu-deps AS llama-server-cpu
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset cpu \
        && cmake --build build/llama-server-cpu -- -l $(nproc) \
        && cmake --install build/llama-server-cpu --component llama-server --strip \
        && for lib in \
            /usr/lib64/libgomp.so* \
            /usr/lib64/libomp.so* \
            /opt/rh/gcc-toolset-13/root/usr/lib64/libgomp.so* \
            /opt/rh/gcc-toolset-13/root/usr/lib64/libomp.so*; do \
                [ -e "$lib" ] && cp -a "$lib" dist/lib/ollama/ || true; \
            done

FROM scratch AS publish-llama-server-cpu
COPY --from=llama-server-cpu dist/lib/ollama /lib/ollama/

FROM cuda-12-deps AS llama-server-cuda_v12
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset llama_cuda_v12_linux \
        && cmake --build build/llama-server-cuda_v12 -- -l $(nproc) \
        && cmake --install build/llama-server-cuda_v12 --component llama-server --strip

FROM scratch AS publish-llama-server-cuda_v12
COPY --from=llama-server-cuda_v12 dist/lib/ollama /lib/ollama/

FROM cuda-13-deps AS llama-server-cuda_v13
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset llama_cuda_v13_linux \
        && cmake --build build/llama-server-cuda_v13 -- -l $(nproc) \
        && cmake --install build/llama-server-cuda_v13 --component llama-server --strip

FROM scratch AS publish-llama-server-cuda_v13
COPY --from=llama-server-cuda_v13 dist/lib/ollama /lib/ollama/

FROM rocm-deps AS llama-server-rocm_v10_0
# No --gcc-toolchain: gcc-toolset-13 is a RHEL Software Collection with no Ubuntu
# equivalent. ROCm's clang uses the image's system GCC (13 on Ubuntu 24.04).
ENV CC=clang CXX=clang++
# Narrow the arch list for a test or dev build without touching the shipped
# default, which stays the preset's full AMDGPU_TARGETS list.
ARG AMDGPU_TARGETS
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset rocm_v10_0_linux ${AMDGPU_TARGETS:+-DAMDGPU_TARGETS=${AMDGPU_TARGETS}} \
        && cmake --build build/llama-server-rocm_v10_0 -- -l $(nproc) \
        && cmake --install build/llama-server-rocm_v10_0 --component llama-server --strip
# Inert on ROCm 10.0, which ships no gfx900/gfx906 at all. Kept because it costs
# nothing and re-fires if a future ROCm reintroduces them.
RUN rm -f dist/lib/ollama/rocm_v10_0/rocblas/library/*gfx90[06]*
# Provenance stamp, read by the preflight toolchain pin (probes.gpu_toolchain).
# TheRock decoupled library SONAMEs from the ROCm release: 10.0.0 ships
# librocblas.so.5.6, which carries no release version, where 7.2.4 shipped
# librocblas.so.5.2.70204 and encoded it. Nothing else in the payload names the
# release either, so without this file a toolchain bump under an unchanged
# payload cannot be detected. NOT an unused artifact -- do not delete.
ARG ROCM_BACKEND_TAG
RUN printf '%s\n' "${ROCM_BACKEND_TAG%%-*}" > dist/lib/ollama/rocm_v10_0/ROCM_VERSION \
    && cat dist/lib/ollama/rocm_v10_0/ROCM_VERSION

FROM scratch AS publish-llama-server-rocm_v10_0
COPY --from=llama-server-rocm_v10_0 dist/lib/ollama /lib/ollama/

FROM vulkan-deps AS llama-server-vulkan
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset vulkan \
        && cmake --build build/llama-server-vulkan -- -l $(nproc) \
        && cmake --install build/llama-server-vulkan --component llama-server --strip

FROM scratch AS publish-llama-server-vulkan
COPY --from=llama-server-vulkan dist/lib/ollama /lib/ollama/

#
# JetPack stages — self-contained with their own base images
#

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK5VERSION} AS jetpack-5
ARG CMAKEVERSION
ARG NINJAVERSION
RUN apt-get update && apt-get install -y curl ccache git unzip \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1 \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux-aarch64.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset llama_cuda_jetpack5 \
        && cmake --build build/llama-server-cuda_jetpack5 -- -l $(nproc) \
        && cmake --install build/llama-server-cuda_jetpack5 --component llama-server --strip

FROM scratch AS publish-llama-server-cuda_jetpack5
COPY --from=jetpack-5 dist/lib/ollama /lib/ollama/

FROM --platform=linux/arm64 nvcr.io/nvidia/l4t-jetpack:${JETPACK6VERSION} AS jetpack-6
ARG CMAKEVERSION
ARG NINJAVERSION
RUN apt-get update && apt-get install -y curl ccache git unzip \
    && curl -fsSL https://github.com/Kitware/CMake/releases/download/v${CMAKEVERSION}/cmake-${CMAKEVERSION}-linux-$(uname -m).tar.gz | tar xz -C /usr/local --strip-components 1 \
    && curl -fsSL -o /tmp/ninja.zip https://github.com/ninja-build/ninja/releases/download/v${NINJAVERSION}/ninja-linux-aarch64.zip \
    && unzip /tmp/ninja.zip -d /usr/local/bin \
    && rm /tmp/ninja.zip
ENV CMAKE_GENERATOR=Ninja
COPY LLAMA_CPP_VERSION .
COPY llama/server llama/server
COPY llama/compat llama/compat
COPY cmake cmake
RUN --mount=type=cache,target=/root/.ccache \
    cmake -S llama/server --preset llama_cuda_jetpack6 \
        && cmake --build build/llama-server-cuda_jetpack6 -- -l $(nproc) \
        && cmake --install build/llama-server-cuda_jetpack6 --component llama-server --strip

FROM scratch AS publish-llama-server-cuda_jetpack6
COPY --from=jetpack-6 dist/lib/ollama /lib/ollama/

#
# MLX stage
#

FROM base AS mlx
ARG CUDA13VERSION=13.0
ARG OLLAMA_MLX_BUILD_JOBS=
ARG OLLAMA_MLX_NVCC_THREADS=2
ARG MLX_CUDA_RAM_MB=
RUN dnf install -y cuda-toolkit-${CUDA13VERSION//./-} \
    && dnf install -y openblas-devel lapack-devel \
    && dnf install -y libcudnn9-cuda-13 libcudnn9-devel-cuda-13 \
    && dnf install -y libnccl libnccl-devel
ENV PATH=/usr/local/cuda-13/bin:$PATH
ENV BLAS_INCLUDE_DIRS=/usr/include/openblas
ENV LAPACK_INCLUDE_DIRS=/usr/include/openblas
ENV CGO_LDFLAGS="-L/usr/local/cuda-13/lib64 -L/usr/local/cuda-13/targets/x86_64-linux/lib/stubs"
WORKDIR /go/src/github.com/ollama/ollama
COPY CMakeLists.txt CMakePresets.json .
COPY cmake cmake
COPY mlx mlx
COPY mlxrunner/xgrammar/native mlxrunner/xgrammar/native
COPY go.mod go.sum .
COPY MLX_VERSION MLX_C_VERSION .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
RUN --mount=type=cache,target=/root/.ccache \
    --mount=type=bind,from=local-mlx,target=/tmp/local-mlx \
    --mount=type=bind,from=local-mlx-c,target=/tmp/local-mlx-c \
    if [ -f /tmp/local-mlx/CMakeLists.txt ]; then \
        export OLLAMA_MLX_SOURCE=/tmp/local-mlx; \
    fi \
    && if [ -f /tmp/local-mlx-c/CMakeLists.txt ]; then \
        export OLLAMA_MLX_C_SOURCE=/tmp/local-mlx-c; \
    fi \
    && cmake -S . -B build/mlx_cuda_v13 -DOLLAMA_MLX_BACKENDS=cuda_v13 -DBLAS_INCLUDE_DIRS=/usr/include/openblas -DLAPACK_INCLUDE_DIRS=/usr/include/openblas -DCMAKE_CUDA_FLAGS="-t ${OLLAMA_MLX_NVCC_THREADS}" ${MLX_CUDA_RAM_MB:+-DMLX_CUDA_RAM_MB=${MLX_CUDA_RAM_MB}} -DOLLAMA_PAYLOAD_INSTALL_PREFIX=/go/src/github.com/ollama/ollama/dist \
        && cmake --build build/mlx_cuda_v13 --target ollama-mlx-cuda_v13 -- -l $(nproc) ${OLLAMA_MLX_BUILD_JOBS:+-j ${OLLAMA_MLX_BUILD_JOBS}}

FROM scratch AS publish-mlx
COPY --from=mlx /go/src/github.com/ollama/ollama/dist/lib/ollama /lib/ollama/

#
# Go build
#

FROM base AS build
WORKDIR /go/src/github.com/ollama/ollama
COPY go.mod go.sum .
RUN curl -fsSL https://golang.org/dl/go$(awk '/^go/ { print $2 }' go.mod).linux-$(case $(uname -m) in x86_64) echo amd64 ;; aarch64) echo arm64 ;; esac).tar.gz | tar xz -C /usr/local
ENV PATH=/usr/local/go/bin:$PATH
RUN go mod download
COPY . .
ARG GOFLAGS="'-ldflags=-w -s'"
ENV CGO_ENABLED=1
ARG CGO_CFLAGS
ARG CGO_CXXFLAGS
ENV CGO_CFLAGS="${CGO_CFLAGS}"
ENV CGO_CXXFLAGS="${CGO_CXXFLAGS}"
RUN --mount=type=cache,target=/root/.cache/go-build \
    go build -trimpath -buildmode=pie -o /bin/ollama .
RUN --mount=type=cache,target=/root/.cache/go-build \
    cmake -S . -B build/go-license \
        -DOLLAMA_LLAMA_BACKENDS= \
        -DOLLAMA_MLX_BACKENDS= \
    && cmake --build build/go-license --target ollama-go-license

FROM scratch AS publish-go
COPY --from=build /bin/ollama /bin/ollama
COPY --from=build /go/src/github.com/ollama/ollama/build/go-license/lib/ollama/GO_LICENSE /lib/ollama/GO_LICENSE

#
# Assembly stages — combine llama-server variants + GPU runtime libs
#

FROM --platform=linux/amd64 scratch AS amd64
COPY --from=llama-server-cpu      dist/lib/ollama /lib/ollama/
COPY --from=llama-server-cuda_v12 dist/lib/ollama /lib/ollama/
COPY --from=llama-server-cuda_v13 dist/lib/ollama /lib/ollama/
COPY --from=llama-server-vulkan   dist/lib/ollama /lib/ollama/
COPY --from=mlx     /go/src/github.com/ollama/ollama/dist/lib/ollama /lib/ollama/

FROM --platform=linux/arm64 scratch AS arm64
COPY --from=llama-server-cpu dist/lib/ollama /lib/ollama/
COPY --from=llama-server-cuda_v12 dist/lib/ollama /lib/ollama/
COPY --from=llama-server-cuda_v13 dist/lib/ollama /lib/ollama/
COPY --from=jetpack-5 dist/lib/ollama/ /lib/ollama/
COPY --from=jetpack-6 dist/lib/ollama/ /lib/ollama/

FROM scratch AS rocm
COPY --from=llama-server-cpu  dist/lib/ollama /lib/ollama
COPY --from=llama-server-rocm_v10_0 dist/lib/ollama /lib/ollama

FROM --platform=linux/amd64 scratch AS amd64-archive
COPY --from=amd64 /lib/ollama /lib/ollama/
COPY --from=llama-server-rocm_v10_0 dist/lib/ollama /lib/ollama/

FROM --platform=linux/arm64 scratch AS arm64-archive
COPY --from=arm64 /lib/ollama /lib/ollama/

FROM ${TARGETARCH}-archive AS archive
COPY --from=build /bin/ollama /bin/ollama
COPY --from=build /go/src/github.com/ollama/ollama/build/go-license/lib/ollama/GO_LICENSE /lib/ollama/GO_LICENSE

FROM ${FLAVOR} AS image-archive
COPY --from=build /bin/ollama /bin/ollama
COPY --from=build /go/src/github.com/ollama/ollama/build/go-license/lib/ollama/GO_LICENSE /lib/ollama/GO_LICENSE

FROM ubuntu:24.04
ARG APT_MIRROR=http://archive.ubuntu.com/ubuntu
ARG APT_PORTS_MIRROR=http://ports.ubuntu.com/ubuntu-ports
RUN sed -i \
        -e "s|http://archive.ubuntu.com/ubuntu|$APT_MIRROR|g" \
        -e "s|http://ports.ubuntu.com/ubuntu-ports|$APT_PORTS_MIRROR|g" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y ca-certificates libvulkan1 libopenblas0 \
    && sed -i \
        -e "s|$APT_MIRROR|http://archive.ubuntu.com/ubuntu|g" \
        -e "s|$APT_PORTS_MIRROR|http://ports.ubuntu.com/ubuntu-ports|g" \
        /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --from=image-archive /bin /usr/bin
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
COPY --from=image-archive /lib/ollama /usr/lib/ollama
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NVIDIA_VISIBLE_DEVICES=all
ENV OLLAMA_HOST=0.0.0.0:11434
EXPOSE 11434
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
