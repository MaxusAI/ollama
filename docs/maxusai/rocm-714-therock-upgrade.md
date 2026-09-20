# ROCm under TheRock: moving the ROCm backend off AlmaLinux, and which version to move to

MaxusAI-fork reference (fork-only; does not exist upstream). Investigated 2026-09-20 against
`main` at `e4caf7d8` (v0.34.2-dynres, `LLAMA_CPP_VERSION=b10969`).

> **Summary.** `rocm/dev-almalinux-8` has no ROCm 7.14 or 10.x tag and will not get one — AMD
> stopped publishing AlmaLinux ROCm images at `7.2.4-complete` (2026-05-28) when ROCm moved to
> TheRock. That finding stands. What changed is the conclusion drawn from it: the ROCm backend
> stage can move to Ubuntu **on its own**, leaving `base` and every other amd64 stage on
> AlmaLinux 8, because the ROCm payload's runtime is not AlmaLinux — the final image is already
> `FROM ubuntu:24.04`. This branch implements that split.
>
> **Recommended target: ROCm 10.0.0, not 7.14.1** (§5). Both exist, both are Ubuntu-only, both
> name gfx1151, and both cost the same to adopt — but only 10.0.0 still has artifacts in AMD's
> package channels, and only 10.0.0 keeps an AlmaLinux escape hatch open.

---

## 1. Target version: 7.14.1 exists — the request was right, the URL was wrong

The original brief reported seeing `rocm-7.14.0` at
[ROCm/legacy-rocm-build](https://github.com/ROCm/legacy-rocm-build/releases) and no 7.14.1, and
asked whether 7.14.1 was a mistake. It is not.

`ROCm/legacy-rocm-build` is the **legacy** release home. ROCm releases moved to
[ROCm/TheRock](https://github.com/ROCm/TheRock) at 7.14 — which is the transition that makes this
job structural rather than cosmetic. Both repositories are real; they carry different things now.

| Repository | Newest ROCm tags | Verified with |
| --- | --- | --- |
| `ROCm/legacy-rocm-build` | `rocm-7.14.0` (2026-07-16), `rocm-7.2.4`, … also `therock-7.9.0` | `gh api repos/ROCm/legacy-rocm-build/releases` |
| `ROCm/TheRock` | `therock-7.14.1` (2026-08-31), `therock-10.0` (2026-08-26), `therock-7.14` (2026-07-15), `therock-7.13` … `7.10` | `gh api repos/ROCm/TheRock/releases` |

**7.14.1 exists** — `therock-7.14.1`, "ROCm Core SDK 7.14.1 Release", published
2026-08-31T17:30:24Z. Nothing was substituted for it. It is simply not the version this document
recommends, for reasons that are about artifact availability rather than about the version number
(§5).

### The release notes, verbatim

From the `rocm-7.14.0` release body:

> ROCm Core SDK 7.14.0 transitions ROCm to [TheRock](https://github.com/ROCm/TheRock), a build
> and release system that introduces a modular architecture to improve flexibility,
> maintainability, and alignment with community use cases:
>
> * **Leaner core**: The Core SDK focuses on essential runtime and development components.
> * **Use case-specific expansions**: Optional domain-specific SDKs for AI, data science, and HPC.
> * **Modular installation**: Install only the components required for your workflow.

and

> ROCm 7.14.0 follows the versioning discontinuity that began with the 7.9.0 preview release.

### gfx1151 is named — but it is weaker evidence than it looks

> ROCm 7.14.0 adds support for the following AMD APUs:
>
> * AMD Ryzen AI MAX+ PRO 495 (gfx1151)
> * AMD Ryzen AI MAX PRO 490 (gfx1151)
> * AMD Ryzen AI MAX PRO 485 (gfx1151)
> * AMD Ryzen AI 5 435 (gfx1153) …

Read carefully, this adds **new PRO-branded SKUs that happen to be gfx1151**, not new gfx1151 ISA
support. gfx1151 is already a first-class target on the toolchain the fork ships today: it is in
`rocm_v7_2_linux`'s `AMDGPU_TARGETS`, the production host is gfx1151 on a `ROCMVERSION=7.2.4`
build, and the 7.2.1 image on this host carries `TensileLibrary_lazy_gfx1151.dat` (§7).

The host here is a Ryzen AI Max+ **395** — the non-PRO part — so none of the three newly named
SKUs is this machine. **"7.14 names gfx1151" is not a load-bearing argument for the upgrade**, and
it does not discriminate between 7.14.1 and 10.0.0, since both support gfx1151.

---

## 2. The constraint: no AlmaLinux ROCm image past 7.2.4

`docker manifest inspect` (resolves a manifest without downloading layers):

| Tag | Result |
| --- | --- |
| `rocm/dev-almalinux-8:7.14.0-complete` | `no such manifest` |
| `rocm/dev-almalinux-8:7.14.1-complete` | `no such manifest` |
| `rocm/dev-almalinux-8:7.14.1-full` | `no such manifest` |
| `rocm/dev-almalinux-8:7.14.1` | `no such manifest` |
| `rocm/dev-almalinux-8:latest` | `no such manifest` |
| `rocm/dev-almalinux-8:7.2.4-complete` | **manifest returned** (control — the probe works) |

The Docker Hub tag listing explains it: `rocm/dev-almalinux-8` has **23 tags**, newest
`7.2.4-complete`, last pushed **2026-05-28** — nothing since, across two ROCm feature releases and
a major version. AlmaLinux dev images were not carried across the TheRock transition.

What TheRock publishes instead, all verified present:

| Image | Relevant tags |
| --- | --- |
| `rocm/dev-ubuntu-22.04` | `7.14.1-full`, `10.0.0-full`, `7.14.0-full`, `latest` |
| `rocm/dev-ubuntu-24.04` | `7.14.1-full`, `10.0.0-full`, `7.14.0-full`, `latest` |
| `rocm/dev-ubuntu-26.04` | `7.14.1-full`, `10.0.0-full`, `7.14.0-full`, `latest` |

Note the suffix change **`-complete` → `-full`**. Because TheRock has already renamed it once, the
Dockerfile change parameterises the whole tag (`ROCM_BACKEND_TAG`) rather than just the version.

---

## 3. The ROCm-stage-only split, and the ABI question

The earlier read of this file said repointing `Dockerfile:17` would drag every amd64 stage onto
Ubuntu. That is true of line 17 — but line 17 does not have to move.

### What `base` is actually for

`base-amd64` is the ROCm image, and `cpu-deps`, `cuda-12-deps`, `cuda-13-deps`, `vulkan-deps`,
`mlx` and `build` all derive `FROM base`. None of them uses ROCm. The ROCm image is the parent of
the whole amd64 tree only because it happened to also be a usable AlmaLinux 8 build environment.

So the split is: give the ROCm backend its own Ubuntu base, and leave line 17 alone.

```
base-amd64 (rocm/dev-almalinux-8:7.2.1-complete, glibc 2.28)   <- UNCHANGED
  └─ base ─┬─ cpu-deps, cuda-12-deps, cuda-13-deps, vulkan-deps, mlx, build

rocm-base (rocm/dev-ubuntu-24.04:10.0.0-full, glibc 2.39)      <- NEW, parents nothing else
  └─ rocm-deps ─ llama-server-rocm_v10_0
```

`base-amd64` keeps using a ROCm image it does not need, which is inherited oddity rather than
something this change introduces. Cleaning that up (`FROM almalinux:8` + gcc-toolset-13, as
`base-arm64` already does) is a separate, worthwhile change and is deliberately **not** bundled
here.

**CI has been doing this for some time already.** `.github/workflows/test.yaml`'s ROCm job runs
in `rocm/dev-ubuntu-22.04` — the backend has been compiled on Ubuntu in CI while being shipped
from AlmaLinux. The split makes the Dockerfile agree with CI rather than introducing something
new. That job's container is bumped here to track `ROCM_BACKEND_TAG`, since otherwise it would
build a payload named `rocm_v10_0` with a 7.2.1 toolchain.

### The ABI question, answered — the concern is inverted

The worry was: ROCm `.so` files built against glibc 2.39 get `COPY --from`'d into a final image
that is AlmaLinux 8 (glibc 2.28), and fail at `dlopen` time as a silent backend-unavailable.

**The final runtime image is not AlmaLinux.** The last stage of the Dockerfile is `FROM ubuntu:24.04`.
Verified empirically against an image this host actually built:

```
$ docker run --rm --entrypoint sh maxusai-ollama:0.34.1-rocm721-85d5734d -c '...'
NAME="Ubuntu"
VERSION_ID="24.04"
ldd (Ubuntu GLIBC 2.39-0ubuntu8.9) 2.39
/usr/lib/ollama/rocm_v7_2
/usr/lib/ollama/rocm_v7_2/librocblas.so.5.2.70201
```

So today the ROCm payload is built at glibc 2.28 and *runs* at 2.39, working by glibc's forward
compatibility. Building it at 2.39 and running it at 2.39 is an **exact match — strictly safer
than the status quo** for the Docker image. For the fork's own gfx1151 deployment, which is
Docker, the split has no ABI cost at all.

### Where the cost actually lands: the published tarball

`amd64-archive` copies the ROCm payload alongside the others, and `release.yaml` builds
`archive-target: archive` and publishes `dist/*.tgz`. That tarball is extracted onto arbitrary
distros, and that is where a glibc floor is real.

Measured floor of the currently shipped payload (`readelf -V` on `libggml-hip.so` extracted from
the built image): highest requirement **`GLIBC_2.27`**, consistent with an AlmaLinux 8 build.

After the split, the amd64 tarball becomes mixed:

| Payload | Built on | glibc floor |
| --- | --- | --- |
| `cpu`, `cuda_v12`, `cuda_v13`, `vulkan`, `mlx` | AlmaLinux 8 | ~2.27 (unchanged) |
| `rocm_v10_0` | Ubuntu 24.04 | 2.39 |

It is not only our own compiled objects. `llama/server/CMakeLists.txt:552` installs a
`RUNTIME_DEPENDENCY_SET` whose `PRE_INCLUDE_REGEXES` covers `drm drm_amdgpu numa elf tinfo`
alongside the ROCm libraries — those are **system** libraries taken from the build image, so the
whole bundled set moves to the Ubuntu floor, not just `libggml-hip.so`.

Consequence, stated plainly: **a tarball user on glibc < 2.39 (RHEL 9 = 2.34, Debian 12 = 2.36,
Ubuntu 22.04 = 2.35) would get a working CPU/CUDA/Vulkan ollama and a ROCm backend that does not
load** — and because backends are discovered by glob and loaded by `dlopen`, that is a silent
"no ROCm device" rather than an error. The split is viable, but this is its price and it should
be an explicit decision, not a side effect.

Three ways to handle it, in order of preference:

1. **Build the ROCm stage on `rocm/dev-ubuntu-22.04:<tag>` instead** — verified to exist for both
   candidate versions, and drops the floor from 2.39 to 2.35. Cheapest mitigation; costs nothing
   but a tag change, since the Docker runtime (Ubuntu 24.04) runs a 22.04-built payload fine.
2. **Accept and document the floor.** Defensible for this fork specifically, whose own deployment
   is Docker-only.
3. **Install ROCm into the existing AlmaLinux 8 base from TheRock's `rhel8/` packages** and drop
   the split entirely. This restores a 2.28 floor — and it is available **only for 10.0.0** (§4),
   which is one of the reasons 10.0.0 is the recommended target.

---

## 4. What TheRock actually distributes

TheRock's portable build image is `FROM quay.io/pypa/manylinux_2_28_x86_64`
(`dockerfiles/build_manylinux_x86_64.Dockerfile`) — **glibc 2.28, the same floor as AlmaLinux 8**.
(`dockerfiles/README.md` claims these images target "glibc 2.39 or greater", which contradicts its
own `FROM` line; the `FROM` line is the fact.) ROCm built by TheRock is therefore *intended* to be
glibc-2.28-compatible, and TheRock still installs to `/opt/rocm` via symlink. An AlmaLinux 8 ROCm
install is not inherently impossible — it just is not what AMD ships as an image.

But AMD's channels carry only the current version per channel, and **7.14 has aged out of all of
them**:

| Channel | `core/tarball/` versions present |
| --- | --- |
| `stable.repo.amd.com` | `10.0.0` only |
| `rc.repo.amd.com` | `10.1.0rc0`, `10.1.0rc1` |
| `nightly.repo.amd.com` | `10.1.0a*` |
| `dev.repo.amd.com` | `10.1.0.dev0+*` |
| `bkc.repo.amd.com` | (empty) |

`stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/` likewise offers `10.0.0` only. **No 7.14
tarball, RPM or wheel is obtainable from any channel.** 7.14.1 exists in exactly one delivery
form: the three Ubuntu Docker images.

By contrast `stable.repo.amd.com/rocm/core/packages/` carries per-distro trees including
**`rhel8/`** — 460 RPMs at 10.0.0, per-architecture, including:

```
amdrocm-blas10.0-gfx1151-10.0.0-4.x86_64.rpm
amdrocm-core-devel10.0-gfx1151-10.0.0-4.x86_64.rpm
```

---

## 5. Recommendation: 10.0.0, not 7.14.1

Both versions cost the same to adopt. Both are Ubuntu-only `-full` images with the same
`/opt/rocm` layout, and the branch in §9 switches between them with one build arg. The
recommendation therefore turns on everything *other* than the migration work.

| | ROCm 7.14.1 | ROCm 10.0.0 |
| --- | --- | --- |
| Exists | yes (`therock-7.14.1`, 2026-08-31) | yes (`therock-10.0`, 2026-08-26) |
| Position in the line | last patch of the 7.x line | current stable; 10.1 already in rc/nightly |
| Docker dev image | ubuntu 22.04 / 24.04 / 26.04 | ubuntu 22.04 / 24.04 / 26.04 |
| Tarballs in any channel | **none** | stable |
| Native packages, incl. `rhel8/` | **none** | stable, with per-gfx packages incl. gfx1151 |
| Python wheels | **none** | stable |
| gfx1151 | supported | supported |

**Why 10.0.0:**

1. **Artifact availability is the decisive axis.** 7.14.1 is reachable only through three Docker
   image tags. If AMD retires them — and they have already stopped updating a whole image family
   this year — the version becomes unobtainable and the build unreproducible. 10.0.0 is reachable
   four ways.
2. **It keeps the AlmaLinux escape hatch open.** Mitigation 3 in §3 — install ROCm into the
   existing AlmaLinux 8 base from `rhel8/` RPMs and delete the split, restoring the 2.28 tarball
   floor — exists **only for 10.0.0**. Choosing 7.14.1 makes the Ubuntu split a one-way door;
   choosing 10.0.0 leaves a documented way back.
3. **Support horizon.** 7.14 is the end of the 7.x line following the 7.9.0 versioning
   discontinuity. Adopting it in September 2026 means adopting a line upstream has already moved
   past, and doing this migration again sooner.
4. **The gfx1151 argument does not discriminate.** Both support it; the 7.14 release-note mention
   is PRO-SKU enablement (§1).

**The honest case for 7.14.1**, since there is one: it is a smaller jump from 7.2.4 — same major
line — so HIP API and codegen surprises should be smaller. That argument is only decisive if
10.0.0 fails to build where 7.14.1 succeeds, which is precisely why 10.0.0 was built first (§8).

---

## 6. Naming: `rocm_v10_0`

`rocm_v7_2` is **not** a "ROCm major 7" bucket:

1. `rocm_v7_1` is a live backend alongside it (`cmake/local.cmake`, `CMakePresets.json`,
   `.github/workflows/test.yaml`). If the name meant "ROCm 7" there could only be one.
2. `cmake/local.cmake` says so outright: *"Keep the backend names versioned so future packaging
   can install side-by-side ROCm payloads without changing the superbuild interface."*
3. `ollama_rocm_preset` hard-fails if `rocm_v7_1` is used off Windows or `rocm_v7_2` on Windows —
   they are distinct platform/version slots, not aliases.

The convention is `rocm_v<major>_<minor>`. So 10.0 is **`rocm_v10_0`**, not `rocm_v10` — with
10.1.0 already in rc and nightly, a bare `rocm_v10` would collide with the next release and defeat
exactly the side-by-side property the comment protects. (7.14.1 would be `rocm_v7_14`.)

**Renaming is safe at runtime**, which was the main thing to get wrong. Payload directories are
discovered by glob, never from a hardcoded list (`discover/runner.go:48`):

```go
files, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
```

and classified by **prefix**, not exact match (`discover/runner.go:554-556`):

```go
func isROCmLibraryDir(name string) bool {
	return strings.HasPrefix(name, "rocm")
}
```

`nativeProbeHasROCm` uses `strings.Contains(base, "rocm")`. Grepping the Go tree found **nothing
that parses a version out of a backend directory name** — it is a pure label. `ml/path.go` only
locates the root. The `"/lib/ollama/rocm_v7_2"` strings in `discover/runner_test.go`,
`discover/native_probe_test.go` and `discover/llama_server_test.go` are **fixture inputs the tests
construct**, not assertions about the shipped payload name; they are deliberately left untouched.

---

## 7. rocBLAS layout and the toolchain pin

`discover/amd.go:116-119` derives supported GPUs by globbing
`<libdir>/rocblas/library/TensileLibrary_lazy_gfx*.dat` — the same tree `Dockerfile` prunes
gfx900/906 from. If TheRock changed that layout, GPU discovery degrades **silently**. Baselines
measured on this host:

| Image | glibc | gcc | `librocblas` SONAME | `TensileLibrary_lazy_*` | gfx1151 |
| --- | --- | --- | --- | --- | --- |
| `rocm/dev-almalinux-8:7.2.1-complete` | 2.28 | gcc-toolset-13 | `librocblas.so.5.2.70201` | 12 files | present |
| `rocm/dev-ubuntu-24.04:7.2.2-complete` | 2.39 | 13.3.0 | `librocblas.so.5.2.70202` | 12 files | present |
| shipped payload (`maxusai-ollama:0.34.1-rocm721-*`) | — | — | `librocblas.so.5.2.70201` | 12 files | present |

The Ubuntu/AlmaLinux pair is the useful control: **changing distro alone changes neither the
Tensile layout nor the SONAME scheme**, so any difference observed on 10.0.0 or 7.14.1 is a
TheRock change and nothing else. Ubuntu 24.04 also ships **gcc 13.3**, i.e. the same compiler
generation as `gcc-toolset-13`, which is why the split drops `--gcc-toolchain` rather than
replacing it.

### The `toolchain_build` pin (PR #355, open — not merged)

`docs/maxusai/vision-suite/preflight/probes.py` gains `gpu_toolchain()`, which globs
`/usr/lib/ollama/rocm*/librocblas.so.*` and decodes the SONAME:

```python
m = re.search(r"librocblas\.so\.\d+\.\d+\.(\d{5,})$", line)
raw = m.group(1)
major, minor, patch = raw[:-4], raw[-4:-2], raw[-2:]
```

Two things follow, both checked rather than assumed:

- **The glob survives the rename.** `rocm*` matches `rocm_v10_0`. No change needed.
- **The arithmetic survives both targets.** Exercised directly: `70201` → `rocm-7.2.1` and
  `70202` → `rocm-7.2.2` (both matching the real files above), `71401` → `rocm-7.14.1`,
  `100000` → `rocm-10.0.0`. The docstring anticipates the last of these.

The parser's failure mode is a **changed scheme**, not a bigger number: `librocblas.so.10.0.0`,
`librocblas.so.5` and any 4-digit micro all return `None`. Whether TheRock keeps the
`so.<X>.<Y>.<NNNNN>` form is the thing to check on the target image (§10).

---

## 8. What was built, and what happened

### Build 1 — the split itself, isolated from the version change

**`docker build --target publish-llama-server-rocm_v10_0 --build-arg ROCM_BACKEND_TAG=7.2.2-complete
--build-arg AMDGPU_TARGETS=gfx1151` → exit 0.**

This deliberately holds the ROCm version near-constant (7.2.2 vs the 7.2.4 in production) and
changes only the distro, so a failure would be attributable to the split rather than to TheRock.
It is the control for every observation below. Narrowing `AMDGPU_TARGETS` to `gfx1151` keeps it to
one architecture instead of the preset's 13.

What it establishes:

- **ggml-hip compiles under ROCm's clang on Ubuntu with no `--gcc-toolchain`.** Ubuntu 24.04's
  system compiler is gcc 13.3.0 — the same generation as `gcc-toolset-13` — so dropping the RHEL
  Software Collection flag is sufficient, not merely tolerable.
- **The payload layout is byte-for-byte the same shape.** 12 `TensileLibrary_lazy_gfx*.dat` files
  including `gfx1151`, and `librocblas.so.5.2.70202`, matching both the AlmaLinux 7.2.1 image and
  the payload shipped in production. `discover/amd.go`'s glob and PR #355's SONAME parser both
  keep working across the distro change.
- **The `AMDGPU_TARGETS` build arg works** without disturbing the shipped default, which stays the
  preset's full list.

### The glibc floor, measured rather than assumed

`readelf -V`, highest required symbol version:

| Artifact | Built on | Max `GLIBC_*` |
| --- | --- | --- |
| `libggml-hip.so`, shipped today | AlmaLinux 8 | **2.27** |
| `libggml-hip.so`, this build | Ubuntu 24.04 | **2.38** |
| whole bundled payload, this build | Ubuntu 24.04 | **2.38** |

So the floor moves **2.27 → 2.38**, not to 2.39: Ubuntu 24.04's glibc is 2.39 but nothing in the
payload actually references a 2.39 symbol. The bundle's provenance is visible directly — it ships
`libelf-0.190.so`, which is Ubuntu 24.04's elfutils, where an AlmaLinux 8 build yields 0.18x. That
confirms §3's point empirically: the floor is set by the bundled *system* libraries
(`drm`, `numa`, `elf`, `tinfo`) as much as by our own objects.

Practically, glibc ≥ 2.38 excludes RHEL 9 (2.34), Ubuntu 22.04 (2.35) and Debian 12 (2.36) for the
**tarball's ROCm payload only**. Mitigation 1 in §3 — building on `rocm/dev-ubuntu-22.04` — would
land around 2.35 and is untested.

### Build 2 — ROCm 10.0.0

> Pending: `rocm/dev-ubuntu-24.04:10.0.0-full` was still downloading when this was written. The
> Dockerfile defaults to it (`ROCM_BACKEND_TAG=10.0.0-full`); the command is build 1 with that arg
> omitted. **Nothing in this document claims ROCm 10.0.0 has been compiled against.** What build 1
> establishes is that the *split* is sound and that any failure on 10.0.0 is attributable to
> TheRock's toolchain, not to the distro move — which is exactly what makes build 2 diagnostic.

---

## 9. What this branch changes

| File | Change |
| --- | --- |
| `Dockerfile` | New `ROCM_BACKEND_IMAGE` / `ROCM_BACKEND_TAG` args; new `rocm-base` (Ubuntu) + `rocm-deps` stages replacing `rocm-7-deps`; `llama-server-rocm_v10_0` drops `--gcc-toolchain` and gains an optional `AMDGPU_TARGETS` build arg; payload dir renamed |
| `llama/server/CMakePresets.json` | `rocm_v7_2_*` → `rocm_v10_0_*`, incl. `OLLAMA_RUNNER_DIR` |
| `cmake/local.cmake` | backend enum, Linux/Windows guards, shared ROCm block; stale "ROCm 7.1 and 7.2" comment corrected |
| `.github/workflows/{test,release,test-llamacpp-update}.yaml` | target, payload and cache-ref renames; the ROCm CI container bumped to `rocm/dev-ubuntu-22.04:10.0.0-full` to track `ROCM_BACKEND_TAG` |
| `docs/development.md`, `docs/maxusai/spec/fast-platform-dev-loops.md` | documented backend values and the dev-loop recipe |

Deliberately **not** changed:

- `Dockerfile:17` and the `base` chain — every non-ROCm amd64 stage keeps its AlmaLinux 8 /
  glibc 2.28 floor.
- The three `discover/*_test.go` fixtures (§6).
- `docs/maxusai/vision-suite/preflight/expectations.toml` — a profile for a build that has not
  been measured would be a fabricated row, which ADR 0011 rule 4 exists to forbid.
- `llama/compat/*` — confirmed unaffected (§10).
- Historical narrative docs that record past builds under their real names.

---

## 10. What a ROCm bump does *not* touch — confirmed, not assumed

### `llama/compat/*.patch`

`grep -rniE '\b(rocm|hip|hipblas|rocblas|amd|amdgpu|gfx[0-9]+)\b'` across every `.patch`,
`.cmake`, `.cpp` and `.h` in `llama/compat/` returns **zero matches**; `compat.cmake` has no
backend conditionals. 001/002/004/005/801 are host-side `tools/mtmd` C++ with no device code.

The one caveat worth stating: **903** (`903-fix-mmq-ids-padding.patch`) patches
`ggml/src/ggml-cuda/mmq.cu`, and ggml-hip compiles the ggml-cuda tree through HIP. It applies
cleanly regardless of toolchain — it is keyed to llama.cpp sources — but "applies" is not
"generates the same kernel". Compat 906 is genuinely gone, retired in `3dade569` ("b10969 ships
upstream's own revert"); the `rocm-0-34-1-dynres` profile still lists it because that profile
describes the *deployed* 0.34.1 build, not `main`.

### Preflight, before PR #355

Profiles resolve on **(platform, version-string regex)** (`preflight.py:91-117`); an unmatched
combination is a hard error, never a default (ADR 0011 rule 5). Nothing else in a profile moves
when only the toolchain changes: the fork version string comes from `git describe`,
`llama_cpp_build` is a source SHA, and every measured field (`ladder`, `budget_*_tokens`,
`image_*_pixels`, `scaling`, `.pinned`) is a property of the `tools/mtmd` image-token pipeline,
which is host-side C++.

So **a toolchain-only build would match and pass `rocm-0-34-1-dynres` whether or not the new
toolchain is any good** — which is exactly the gap PR #355 closes, and why that PR should land
before this one. The profile's own notes already concede the shape of the blindness:

> Every ladder and budget in this profile would still pass on a 906-less build while vision
> quality collapsed (scene IoU 0.065 on qwen3.8). The ladders here are necessary and not
> sufficient; the vision probes are what catch it.

**Open ADR question, not resolved here.** With PR #355 landed, is a toolchain-only change a *new
profile* or a new `toolchain_build` value on the existing one? README step 1 requires a new profile
for "a genuinely different payload, or a new platform", and a recompiled-kernels-same-sources build
is a different binary payload from the same source payload — a case the rules do not name. My
reading is that it should be a new profile, because the toolchain is part of what produced the
binary. That is an ADR amendment, not a judgement call for this PR.

---

## 11. Should this be gated?

**Yes — `docs/maxusai/amd-upgrade-gate.md` already says so in as many words:**

> Treat a base-image bump as a payload change, not a version bump.

This change adds a base image and moves the ROCm backend onto it, so the gate's trigger is met on
its face. The 2026-09-19 decision lifted the gate for a specific upstream payload move
(`0.32.1-dynres` → `0.34.1-dynres`, b9888 → b10864 + compat 906); it did not repeal the trigger,
and it did not contemplate the compiler changing underneath a fixed payload.

The substantive argument is stronger than the textual one. The gate exists because
`0.32.5-gemma4budget-4259c191` produced **degenerate output** on gfx1151 and had to be rolled
back — a silent numerical-quality failure that plumbing checks did not catch. A ROCm compiler and
rocBLAS bump is the same class of risk through a different door: identical llama.cpp sources,
different generated kernels, different Tensile kernel selection for gfx1151.

Clauses 1 and 2 are issue-specific and do not transfer; clause 5 was recorded as "waived — the
check does not exist". The two that do transfer, and that I would require:

- **Clause 3 analogue** — `--direct-io` / `OLLAMA_IGPU_DIRECT_IO` revalidated on the new toolchain
  as a load-path integrity check, since it was validated under (c) on 7.2.4 specifically.
- **Clause 4 as written** — a vision A/B of **≥6 consecutive rows per build** on `qwen35moe`
  showing **0 degenerate** on the candidate, rollback boundary controlled. This is the clause that
  caught the original failure and it transfers unchanged.

---

## 12. Must be measured before this could ship

The current deployment is green on the vision suite, OCRBench and preflight. None of that
transfers across a toolchain change, because none of it was keyed on the toolchain (§10).

1. **Land PR #355 first**, then add a profile that declares `toolchain_build` for the candidate —
   generated and copied whole. **No existing row may be edited to make the candidate green**
   (ADR 0011 rule 4: *"Copying another profile's numbers across to make a run green is the failure
   this file exists to prevent."*).
2. **The gate's clause-4 vision A/B** — ≥6 consecutive rows on `qwen35moe`, 0 degenerate, rollback
   boundary controlled (§11).
3. **OCRBench** against the current deployment as baseline. A toolchain change can move scores
   without moving any token count.
4. **The fine-text probe**, against the deltas recorded at the 2026-09-19 promotion (`nemotron3`
   9px −1, `qwen3.6` 7px −1, `qwen3.8` 9px +1, reproducible at N=5) — that is the baseline, not
   zero.
5. **gfx1151 GPU discovery explicitly** — that `rocblasGFXTargets` still resolves `gfx1151` from
   the new payload (§7). Cheap to check, silent if it breaks.
6. **A full-arch build.** The build in §8 narrows `AMDGPU_TARGETS`; the shipped preset builds 13
   architectures, and whether `gfx908:xnack-` / `gfx90a:xnack±` are still accepted by the new
   compiler is unverified.
7. **The tarball glibc decision** (§3) — pick mitigation 1, 2 or 3 deliberately, and if the floor
   moves, say so in the release notes.
8. **The ROCm CI job.** Its container is bumped but nothing on this branch ran GitHub Actions.
   `extra-packages: rocm-libs` may be redundant or nonexistent under TheRock; the inline comment
   says to drop that line rather than pin the container back.
9. **Runtime load on real hardware.** Nothing here has been run on the gfx1151 GPU; production
   `ollama-rocm` was not touched at any point.
