# ADR 0042: the fork's ROCm images build on AMD's Ubuntu 24.04 ROCm images, never on AlmaLinux

- **Status:** accepted 2026-09-24, **until further notice** (Glenn: build from
  `rocm/dev-ubuntu-24.04:7.2.4-complete` and `rocm/dev-ubuntu-24.04:10.0.0-full` "until
  further notice"; "no rocm/dev-almalinux-8 in our fork extra scripts"). It changes on his
  word only. Sits beside [ADR 0040](0040-rocm-10-is-experimental-until-it-is-faster.md), which
  keeps ROCm 10 experimental, and [ADR 0032](0032-fork-version-identity-tags-each-upstream-fold.md),
  whose version stamping it leaves unchanged.
- **Date:** 2026-09-24
- **Deciders:** MaxusAI fork maintainers

## Context

Upstream's `Dockerfile` bases **every** amd64 stage — CPU, CUDA, Vulkan, the Go build — on
`rocm/dev-almalinux-8:${ROCMVERSION}-complete`, with `ROCMVERSION` defaulting to 7.2.1. That is
unchanged in v0.34.3, v0.34.4 and upstream `main` as of 2026-09-23. The fork built its gfx1151
images from that file: `0.34.1-dynres-16649e8c` was a full `FLAVOR=rocm` build with
`ROCMVERSION=7.2.4` ([amd-upgrade-gate.md](../amd-upgrade-gate.md), Status), and
`0.34.2-dynres-f67b1aef`, production since 2026-09-21, ships the same 7.2.4 runtime
(`libamdhip64.so.7.2.70204`).

Two facts end that recipe here:

- **AMD has published no AlmaLinux ROCm image since 7.2.4.** The newest of the 23 tags on
  `rocm/dev-almalinux-8` is `7.2.4-complete`, pushed 2026-05-28, and there is no 10.x tag
  (Docker Hub, checked 2026-09-24). Upstream's recipe cannot follow the ROCm line past 7.2.4 at
  all; the ROCm 10 investigation (#359) had to move its ROCm stage onto `rocm/dev-ubuntu-24.04`
  to build 10.0.0.
- **The 7.2.4 AlmaLinux base was a multi-hour pull.** Building the v0.34.3 fold candidate the
  upstream way on 2026-09-24, BuildKit found `rocm/dev-almalinux-8:7.2.4-complete` missing from
  the local store and began pulling it at about 1 MB/s per layer. The build was stopped. The
  Ubuntu 10.0.0 image was already local; the Ubuntu 7.2.4 image was pulled instead
  (4 layers, 6.89 GiB compressed).

A fallback tried in between — a Go-only overlay whose binary came from upstream's `build`
stage on the cached `rocm/dev-almalinux-8:7.2.1-complete` — was discarded under the rule below.

## Decision

1. **Two images, one per ROCm toolchain.** Until further notice, a fork ROCm build uses exactly:

   | toolchain | image | preset | status |
   |---|---|---|---|
   | `rocm7` | `rocm/dev-ubuntu-24.04:7.2.4-complete` | `rocm_v7_2` | production |
   | `rocm10` | `rocm/dev-ubuntu-24.04:10.0.0-full` | `rocm_v10_0` | experimental, [ADR 0040](0040-rocm-10-is-experimental-until-it-is-faster.md) |

   Both stay pulled on the gfx1151 host and are not pruned.
2. **No `rocm/dev-almalinux-8` in anything the fork owns** — scripts, Dockerfiles, harness
   helpers, runbooks' commands. `Dockerfile.rocm` refuses an AlmaLinux image in its first stage.
3. **The fork builds ROCm through its own file.** [`Dockerfile.rocm`](../../../Dockerfile.rocm),
   driven by [`scripts/build_rocm.sh`](../../../scripts/build_rocm.sh), puts every stage on the
   chosen Ubuntu image and produces upstream's `FLAVOR=rocm` image stage for stage. Upstream's
   `Dockerfile` is left exactly as upstream ships it; the fork neither edits it nor builds ROCm
   images from it. That is the same reason `Dockerfile.gemma4budget` and `Dockerfile.applearm`
   are separate files: an edit to upstream's file is a conflict in every fold that touches it.
4. **The payload names its toolchain.** Beside the ROCm payload, `ROCM_VERSION` holds the
   release (preflight's toolchain pin reads it) and `ROCM_IMAGE` the exact image; the image
   carries the label `org.maxusai.rocm.image`.
5. **Version stamping does not change.** `scripts/build_rocm.sh` stamps through
   `scripts/env.sh`, as ADR 0032 requires of every build.
6. **Every build cache is on, for both toolchains** (Glenn, 2026-09-24: "in case we need it
   again"). ccache for the CPU and ROCm compiles; apt `.debs` and lists, one pair per ROCm image;
   the ccache/cmake/ninja tarballs and the Go toolchain; one pristine llama.cpp checkout per
   `LLAMA_CPP_VERSION`, copied to the path `FetchContent` would clone to so compile commands and
   ccache keys match a network build's; the Go module and build caches. All are BuildKit cache
   mounts on the build host. Measured on the toolchain stage: 248 s cold, 8–10 s once its layer is
   invalidated with the caches warm — `Need to get 0 B/5315 kB of archives`, no tarball fetched.
   **They are deleted by `docker builder prune` unless it is run with `--filter type=regular`**,
   and `--no-cache` bypasses them.

## What changes in the payload, and why the first build is measured

The ROCm release is the same — 7.2.4, `librocblas.so.5.2.70204` — but everything around it moves:
ROCm's clang compiles against the system GCC 13.3's libstdc++ instead of `gcc-toolset-13`'s,
glibc is 2.39 instead of 2.28, the bundled ROCm runtime comes from AMD's Ubuntu packages instead
of its RHEL ones, and the CPU ggml variants and the Go binary are built by GCC 13.3 on glibc 2.39.

**Preflight cannot see any of that.** `toolchain_build = "rocm-7.2.4"` passes for an
AlmaLinux-built and an Ubuntu-built payload alike; `ROCM_IMAGE` exists so the difference is at
least on record. The first Ubuntu-built `rocm7` image is therefore measured against the
AlmaLinux-built production image — clause 4 of the [AMD gate](../amd-upgrade-gate.md) and the
OCRBench slice — before it serves anything. For v0.34.3 that measurement is the fold's gfx1151
regression run ([MaxusAI/ollama#372](https://github.com/MaxusAI/ollama/pull/372), recorded in its task doc).

The runtime stage is `ubuntu:24.04` in both recipes, so glibc 2.39 costs nothing inside the
container. Upstream keeps its build stages on glibc 2.28 for the tarballs it publishes; the
fork publishes none (its release workflow fails fast), so that floor binds nothing here.

## Options considered

- **Keep building `FLAVOR=rocm` from upstream's `Dockerfile` with `ROCMVERSION=7.2.4`.** Needs
  the AlmaLinux image, which stops at 7.2.4 and was a multi-hour pull on this host. It has no
  route to ROCm 10 at all.
- **Edit upstream's `Dockerfile` so its ROCm stage builds on Ubuntu**, as #359 does for ROCm 10.
  It works, but it carries a conflict into every fold that touches the file, and it keeps
  AlmaLinux for the file's other stages, which the rule excludes from fork builds.
- **Go-only overlays onto a published image.** Valid only while no native input moves, and the
  Go build still needs a toolchain stage, which in upstream's recipe is AlmaLinux.

## Consequences

- **A fold now has one more thing to compare.** When upstream changes its ROCm stages — the
  presets, the install component, the `rocblas` pruning line, or the runtime stage —
  `Dockerfile.rocm` must be brought level in the same fold. The fold's task doc records the
  check, and the check is a diff of those stages between the old and new tag.
- **Retirement.** `Dockerfile.rocm` retires when upstream builds ROCm on an image AMD still
  publishes and the two recipes are shown equivalent, or when Glenn lifts the rule. It is on
  the [retirement register](../retirement-register.md) under fork tooling.
- **History stays as written.** Images built before 2026-09-24, including the production
  `0.34.2-dynres-f67b1aef`, are AlmaLinux-built; the documents that describe them keep saying so.
- **`rocm10` builds from `main`.** Its `rocm_v10_0` presets sit beside `rocm_v7_2` rather than
  replacing them, as #359 does, so one tree builds both toolchains; and
  `llama/server/CMakeLists.txt` bundles the libraries ROCm 10 split out of 7.2 (`rocm_kpack`,
  `origami`, `clang-cpp`, `LLVM`, `rocm_sysdeps`), without which `libggml-hip` cannot load — names
  that match nothing on ROCm ≤ 7.2. Both are small additive edits to upstream files, on the
  retirement register. Production is unaffected: ROCm 10 stays experimental (ADR 0040).
