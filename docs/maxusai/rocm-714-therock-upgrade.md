# ROCm 7.14 / TheRock: why the toolchain bump is not a version-string change

MaxusAI-fork reference (fork-only; does not exist upstream). Investigated 2026-09-20 against
`main` at `e4caf7d8` (v0.34.2-dynres, `LLAMA_CPP_VERSION=b10969`).

> **Conclusion: blocked, and not by a small margin.** `rocm/dev-almalinux-8` — the image
> `Dockerfile:17` builds every amd64 stage on — has **no ROCm 7.14 tag and never will**. The
> repository stopped at `7.2.4-complete` on 2026-05-28. Under TheRock, AMD publishes the ROCm
> dev images for Ubuntu only. There is therefore no version string that can be put into
> `ARG ROCMVERSION` to produce a 7.14 build, and this branch deliberately contains **no
> Dockerfile change**. What follows is the evidence, the three real migration paths, and the
> reason the most attractive of them points at ROCm **10.0.0**, not 7.14.

---

## 1. Target version: 7.14.1 exists — the request was right, the URL was wrong

The prompt for this investigation reported seeing `rocm-7.14.0` at
[ROCm/legacy-rocm-build](https://github.com/ROCm/legacy-rocm-build/releases) and no 7.14.1,
and asked whether 7.14.1 was a mistake. It is not.

`ROCm/legacy-rocm-build` is, as the name says, the **legacy** release home. ROCm releases moved
to [ROCm/TheRock](https://github.com/ROCm/TheRock) at 7.14, which is exactly the transition that
makes this job large. Both repositories are real and both are current; they just carry different
things now.

| Repository | Newest ROCm tags | Verified |
| --- | --- | --- |
| `ROCm/legacy-rocm-build` | `rocm-7.14.0` (2026-07-16), `rocm-7.2.4`, … also `therock-7.9.0` | `gh api repos/ROCm/legacy-rocm-build/releases` |
| `ROCm/TheRock` | `therock-7.14.1` (2026-08-31), `therock-10.0` (2026-08-26), `therock-7.14` (2026-07-15), `therock-7.13`, `7.12`, `7.11`, `7.10` | `gh api repos/ROCm/TheRock/releases` |

So **7.14.1 exists** — as `therock-7.14.1`, "ROCm Core SDK 7.14.1 Release", published
2026-08-31T17:30:24Z — and it is the correct target for "the ROCm 7.14 line". `legacy-rocm-build`
shows 7.14.0 only because it is no longer where point releases land. Nothing was substituted:
this document targets **7.14.1**, and where 7.14.0 is named it is because a fact was only
verifiable against 7.14.0's release notes.

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

The release notes do name gfx1151 explicitly:

> ROCm 7.14.0 adds support for the following AMD APUs:
>
> * AMD Ryzen AI MAX+ PRO 495 (gfx1151)
> * AMD Ryzen AI MAX PRO 490 (gfx1151)
> * AMD Ryzen AI MAX PRO 485 (gfx1151)
> * AMD Ryzen AI 5 435 (gfx1153) …

Read carefully, this adds **new PRO-branded SKUs that happen to be gfx1151**, not new gfx1151
ISA support. gfx1151 is already a first-class target on the toolchain this fork ships today:
`llama/server/CMakePresets.json` lists `gfx1151` in `rocm_v7_2_linux`'s `AMDGPU_TARGETS`, and the
production deployment is a gfx1151 host running a `ROCMVERSION=7.2.4` build
(`docs/maxusai/amd-upgrade-gate.md`, and the 2026-09-19 promotion recorded there).

The host here is a Ryzen AI Max+ **395** — the non-PRO part — so none of the three newly named
SKUs is this machine. **I would not carry "7.14 names gfx1151" into a PR as the argument for the
upgrade.** It is true and it is quotable, but it describes SKU enablement on an ISA that already
works. If there is a real case for moving, it has to be made from measured kernel/rocBLAS
behaviour on gfx1151, which is precisely what has not been measured (§6).

---

## 2. The blocker: the base image does not exist

`Dockerfile:17`:

```dockerfile
FROM --platform=linux/amd64 rocm/dev-almalinux-8:${ROCMVERSION}-complete AS base-amd64
```

Probed with `docker manifest inspect`, which resolves the manifest without downloading layers
(no image was pulled and no image was built at any point in this investigation):

| Tag | Result |
| --- | --- |
| `rocm/dev-almalinux-8:7.14.0-complete` | `no such manifest` |
| `rocm/dev-almalinux-8:7.14.1-complete` | `no such manifest` |
| `rocm/dev-almalinux-8:7.14.1-full` | `no such manifest` |
| `rocm/dev-almalinux-8:7.14.1` | `no such manifest` |
| `rocm/dev-almalinux-8:latest` | `no such manifest` |
| `rocm/dev-almalinux-8:7.2.4-complete` | **manifest returned** (control — the probe works) |

The Docker Hub tag listing explains why. `rocm/dev-almalinux-8` has **23 tags total**, newest
`7.2.4-complete`, last pushed **2026-05-28**. There is no 7.14 tag, no 10.x tag, and no `-full`
tag. The repository has had nothing pushed to it in the ~4 months since, across two ROCm feature
releases and a major version. AlmaLinux dev images were not carried across the TheRock
transition.

What TheRock does publish, all verified to exist:

| Image | Tags present |
| --- | --- |
| `rocm/dev-ubuntu-22.04` | `7.14.1-full`, `10.0.0-full`, `7.14.0-full`, `latest` (2026-08-31) |
| `rocm/dev-ubuntu-24.04` | `7.14.1-full`, `10.0.0-full`, `7.14.0-full`, `latest` (2026-08-31) |
| `rocm/dev-ubuntu-26.04` | `7.14.1-full`, `10.0.0-full`, `7.14.0-full`, `latest` (2026-08-31) |

Note the **tag suffix changed from `-complete` to `-full`**. Even on a distro that is still
published, `${ROCMVERSION}-complete` would not resolve. Compressed size of
`rocm/dev-ubuntu-24.04:7.14.1-full` is 7.9 GB, so the decompressed image is in the same
~20–26 GB class as the current base — the disk constraint that forbids pulling it is unchanged.

---

## 3. Why this is not a one-line swap to Ubuntu

`base-amd64` is not the ROCm stage. It is the **build OS for every amd64 stage in the file**:

```
Dockerfile:17   base-amd64  ← rocm/dev-almalinux-8
Dockerfile:29   FROM base-${TARGETARCH} AS base
Dockerfile:65   FROM base AS cpu-deps
Dockerfile:69   FROM base AS cuda-12-deps
Dockerfile:74   FROM base AS cuda-13-deps
Dockerfile:79   FROM base AS rocm-7-deps
Dockerfile:82   FROM base AS vulkan-deps
Dockerfile:228  FROM base AS mlx
Dockerfile:270  FROM base AS build
```

Repointing line 17 at `rocm/dev-ubuntu-24.04:7.14.1-full` therefore changes the OS under the CPU,
CUDA 12, CUDA 13, Vulkan, MLX and Go stages as well. Concretely it breaks:

- **Package manager.** `dnf install` / `yum-config-manager` at lines 18–19, 54, 66, 71, 76, 233–236 — Ubuntu has
  neither.
- **CUDA repositories.** The RHEL 8 repo URLs (`.../cuda/repos/rhel8/x86_64/`) must become
  `ubuntu2404`, for both CUDA 12.8 and CUDA 13.0.
- **Compiler toolchain.** `gcc-toolset-13` is a RHEL Software Collection. It has no Ubuntu
  equivalent, and `ENV PATH=/opt/rh/gcc-toolset-13/root/usr/bin:$PATH` plus
  `CXXFLAGS=--gcc-toolchain=/opt/rh/gcc-toolset-13/root/usr` (`Dockerfile:151`) both become
  meaningless.
- **glibc floor of everything we ship.** AlmaLinux 8 is glibc 2.28; Ubuntu 24.04 is glibc 2.39.
  The amd64 tarballs this repo produces would stop running on any distro older than Ubuntu 24.04 /
  RHEL 10 — a user-visible portability regression that has nothing to do with ROCm, affecting the
  CPU and CUDA payloads too.

That last point is the one that makes "just use the Ubuntu image" the wrong answer rather than
merely a big diff.

---

## 4. The glibc-preserving path exists — but it leads to 10.0.0, not 7.14

TheRock's own portable build image is built `FROM quay.io/pypa/manylinux_2_28_x86_64`
(`dockerfiles/build_manylinux_x86_64.Dockerfile`). `manylinux_2_28` **is** glibc 2.28 — the same
floor as AlmaLinux 8. (The prose in `dockerfiles/README.md` claims these images "support building
binaries compatible with most Linux distributions that use glibc 2.39 or greater", which
contradicts its own `FROM` line; the `FROM` line is the fact.) ROCm built by TheRock is therefore
*intended* to be glibc-2.28-compatible, and an AlmaLinux 8 base is not inherently incompatible
with it. TheRock also still installs to `/opt/rocm` via symlink (`install_rocm_tarball.sh`), so
`rocm-7-deps`' `ENV PATH=/opt/rocm/llvm/bin:…` survives in principle.

So the shape of a migration that keeps the current base OS is: keep `FROM almalinux:8`, and
install ROCm from TheRock's per-distro packages or tarballs instead of consuming a prebuilt
"complete" image.

**This does not work for 7.14.** AMD's TheRock package channels carry only the current version
per channel, and 7.14 has already aged out of all of them:

| Channel | `core/tarball/` versions present |
| --- | --- |
| `stable.repo.amd.com` | `10.0.0` only |
| `rc.repo.amd.com` | `10.1.0rc0`, `10.1.0rc1` |
| `nightly.repo.amd.com` | `10.1.0a*` |
| `dev.repo.amd.com` | `10.1.0.dev0+*` |
| `bkc.repo.amd.com` | (empty) |

`stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/` likewise offers `10.0.0` only. **No 7.14
tarball, package or wheel is obtainable from any channel.** On today's evidence, ROCm 7.14.1 is
retrievable *only* as the three Ubuntu Docker images — which is the one delivery form that is
incompatible with keeping the AlmaLinux base.

Meanwhile `stable.repo.amd.com/rocm/core/packages/` does have per-distro trees including
**`rhel8/`** — 460 RPMs at 10.0.0, and they are per-arch, including:

```
amdrocm-blas10.0-gfx1151-10.0.0-4.x86_64.rpm
amdrocm-core-devel10.0-gfx1151-10.0.0-4.x86_64.rpm
```

So the AlmaLinux-8-preserving, gfx1151-supporting, glibc-2.28-preserving path **exists and is
fully packaged — for ROCm 10.0.0**. If the actual goal is "a newer ROCm toolchain on gfx1151"
rather than "the string 7.14 specifically", 10.0.0 is both newer and dramatically cheaper to
adopt, because it is the only option that does not force a base-OS migration.

That is a recommendation, not a decision, and it is out of scope for the request as written.
It is recorded here because a PR proposing 7.14 should not be opened without it.

### Version landscape, for the record

`therock-10.0` (10.0.0) was published 2026-08-26; `therock-7.14.1` was published 2026-08-31,
*after* it. 7.14 is the last of the 7.x line following the 7.9.0 versioning discontinuity, and
10.0 is its successor, already stable, with 10.1 in rc and nightly. Adopting 7.14 in
September 2026 means adopting a line that upstream has already moved past.

---

## 5. Where the version and the `rocm_v7_2` identifier are encoded

Every occurrence found in the tree. **Nothing in this table was changed on this branch** — the
column records what a 7.14 change *would* have to touch.

| Location | What it encodes | Would need to change? |
| --- | --- | --- |
| `Dockerfile:5` | `ARG ROCMVERSION=7.2.1` | Yes — but there is no valid 7.14 value for the image on line 17 |
| `Dockerfile:17` | `rocm/dev-almalinux-8:${ROCMVERSION}-complete` | Yes — image **and** `-complete`→`-full` suffix |
| `Dockerfile:79` | `FROM base AS rocm-7-deps` | No — "rocm-7" is a deps bucket, still accurate for 7.14 |
| `Dockerfile:150,157-160,162-163,315,319` | `llama-server-rocm_v7_2` stage, `rocm_v7_2_linux` preset, rocBLAS prune, final copies | Yes, if the payload dir is renamed |
| `llama/server/CMakePresets.json:201-228, 322-330` | `rocm_v7_2_base/_linux/_user_arch`, `OLLAMA_RUNNER_DIR`, `AMDGPU_TARGETS` | Yes, if renamed |
| `cmake/local.cmake:10, 391-401, 737-741` | backend enum, Linux/Windows guard, shared ROCm build block | Yes, if renamed |
| `.github/workflows/release.yaml:515,622,640` | build target + two registry cache refs | Yes, if renamed |
| `.github/workflows/test.yaml:107-110` | superbuild target/dir/args, `expected_payload` path | Yes, if renamed |
| `.github/workflows/test-llamacpp-update.yaml:106-107,230` | publish target, payload name, `copy_payload` | Yes, if renamed |
| `docs/development.md:42,51` | documented backend values | Yes, if renamed |
| `docs/maxusai/spec/fast-platform-dev-loops.md:55-79` | `rocm_v7_2_user_arch` dev-loop recipe | Yes, if renamed |
| `scripts/env.sh:23` | passes `ROCMVERSION` through | No — generic |
| `discover/runner_test.go:330,376`, `discover/native_probe_test.go:100`, `discover/llama_server_test.go:89` | `"/lib/ollama/rocm_v7_2"` as test fixture strings | **No** — see §5.2 |
| `ml/path.go:18` | doc comment naming `rocm_v7_2` as an example | Cosmetic only |

### 5.1 The naming decision: a 7.14 toolchain wants `rocm_v7_14`, not `rocm_v7_2`

`rocm_v7_2` is **not** a "ROCm major 7" bucket. Three independent pieces of evidence:

1. **A second bucket already exists.** `rocm_v7_1` is a live backend alongside `rocm_v7_2`
   (`cmake/local.cmake:10`, `CMakePresets.json:177-199`, `.github/workflows/test.yaml:211-214`).
   If the name meant "ROCm 7" there could only be one.
2. **The code says so explicitly.** `cmake/local.cmake:738-740`:

   > ROCm 7.1 and 7.2 currently share build settings. Keep the backend names versioned so future
   > packaging can install side-by-side ROCm payloads without changing the superbuild interface.

   The minor version in the name is deliberate, and the stated purpose is side-by-side payloads.
3. **They are already load-bearing as distinct buckets.** `ollama_rocm_preset` hard-fails if
   `rocm_v7_1` is used off Windows or `rocm_v7_2` on Windows — they are separate platform/version
   slots, not aliases.

A 7.14 toolchain is a different minor version producing a different compiled payload, so by the
repo's own stated convention it earns **`rocm_v7_14`**. Reusing `rocm_v7_2` for a 7.14 build would
make the directory name lie about its contents and forfeit the side-by-side property the comment
is protecting.

### 5.2 Renaming is safe at runtime — verified, and this was the main thing to get wrong

Backend payload directories are **discovered by glob, never from a hardcoded list**
(`discover/runner.go:48`):

```go
files, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "*", "*ggml-*"))
```

and a directory is treated as ROCm by **prefix**, not by exact match
(`discover/runner.go:554-556`):

```go
func isROCmLibraryDir(name string) bool {
	return strings.HasPrefix(name, "rocm")
}
```

`nativeProbeHasROCm` (`discover/native_probe_platform.go:105-108`) similarly uses
`strings.Contains(base, "rocm") || strings.Contains(base, "hip")`. `ml/path.go` only locates the
*root*; it does not enumerate backends.

Therefore `rocm_v7_14` is discovered and classified correctly with no Go change. The
`"/lib/ollama/rocm_v7_2"` strings in `discover/runner_test.go`, `discover/native_probe_test.go`
and `discover/llama_server_test.go` are **fixture inputs constructed by the tests**, not
assertions that the shipped payload is named `rocm_v7_2`; they would keep passing untouched and
should be left alone.

The one real runtime coupling to the payload's *internal* layout is
`discover/amd.go:116-119`, which derives supported GPUs by globbing

```
<libdir>/rocblas/library/TensileLibrary_lazy_gfx*.dat
```

This is the same path `Dockerfile:160` prunes (`rm -f …/rocm_v7_2/rocblas/library/*gfx90[06]*`).
Whether TheRock's rocBLAS still ships `TensileLibrary_lazy_gfx*.dat` under
`rocblas/library/` is **unverified and unverifiable without building** — and if it does not, GPU
discovery degrades silently rather than failing loudly. This is the single highest-risk unknown
in the whole migration and is called out again in §7.

---

## 6. What a ROCm bump does *not* touch — confirmed, not assumed

### `llama/compat/*.patch` is unaffected

`grep -rniE '\b(rocm|hip|hipblas|rocblas|amd|amdgpu|gfx[0-9]+)\b'` across every `.patch`,
`.cmake`, `.cpp` and `.h` in `llama/compat/` returns **zero matches**. `compat.cmake` contains no
backend conditionals at all. Patches 001/002/004/005/801 are host-side `tools/mtmd` C++ with no
device code.

One caveat worth stating rather than glossing: **903** (`903-fix-mmq-ids-padding.patch`) patches
`ggml/src/ggml-cuda/mmq.cu`, and ggml-hip compiles the ggml-cuda tree through HIP — so 903 is
device code that a ROCm compiler change *does* recompile, even though the patch contains no
ROCm-specific text. It will still apply (it is keyed to llama.cpp sources, not the toolchain),
but "applies" is not "generates the same kernel". Compat 906 is genuinely gone from the tree,
retired in `3dade569` ("b10969 ships upstream's own revert") — the `rocm-0-34-1-dynres` profile
still lists it because that profile describes the *deployed* 0.34.1 build, not `main`.

### The preflight harness has no ROCm-toolchain axis at all

Profiles resolve on **(platform, version-string regex)** — `preflight.py:91-117` filters on exact
`platform`, then takes the first profile whose `version_pattern` matches the server's reported
version; an unmatched combination is a hard error, never a default (ADR 0011 rule 5).

Searching the whole harness for `ROCMVERSION|rocm724|7\.2\.4|hipblas|rocblas` finds **nothing but
prose**. There is no `rocm_build` pin analogous to `llama_cpp_build` / `mlx_build`.

Consequently, if only the ROCm toolchain changes:

- `platform`, `version_pattern`, `patchset`, `llama_cpp_build`, `expect_patch_marker`, `arches` —
  all unchanged. The fork version string comes from `git describe` and does not encode
  `ROCMVERSION`.
- Every measured field (`ladder`, `budget_*_tokens`, `image_*_pixels`, `scaling`, `.pinned`) is a
  property of the image-token pipeline owned by 001/002/004/005 in `tools/mtmd`, which is host-side
  C++ — expected unchanged.

**So a ROCm-toolchain-only build would match the existing `rocm-0-34-1-dynres` profile and pass it,
whether or not the new toolchain is any good.** That is not a reason to edit a row — ADR 0011 rule
4 forbids exactly that — it is a statement that preflight is structurally blind to this change
class. The profile's own notes already concede the general shape of this blindness:

> Every ladder and budget in this profile would still pass on a 906-less build while vision
> quality collapsed (scene IoU 0.065 on qwen3.8). The ladders here are necessary and not
> sufficient; the vision probes are what catch it.

**Recommended (not done here, because it should not be invented alongside a blocked upgrade):**
add an optional `rocm_build` field to the profile schema, asserted the way `llama_cpp_build` is,
so that a toolchain change is forced to declare itself. Without it the answer to "does this change
need a new profile?" is genuinely undefined: README step 1 says a new profile is required for "a
genuinely different payload, or a new platform", and a recompiled-kernels-same-sources build is a
different binary payload from the same source payload — a case the rules do not currently name.
My reading is that it **should** be a new profile, because the ROCm toolchain is part of what
produced the binary, but that is an ADR-level decision and should be made by amendment rather than
by one PR's judgement call.

---

## 7. Should a ROCm toolchain bump be gated like a payload change?

**Yes — and `docs/maxusai/amd-upgrade-gate.md` already says so in as many words:**

> Treat a base-image bump as a payload change, not a version bump.

A ROCm change *is* a base-image bump — it is literally the image on `Dockerfile:17` — so the gate's
own trigger condition is met on its face. The 2026-09-19 decision lifted the gate for a specific
upstream payload (`0.32.1-dynres` → `0.34.1-dynres`, b9888 → b10864 + compat 906); it did not
repeal the trigger, and it did not contemplate the compiler changing underneath a fixed payload.

The substantive argument is stronger than the textual one. The gate exists because
`0.32.5-gemma4budget-4259c191` produced **degenerate output** on gfx1151 and had to be rolled
back — a silent numerical-quality failure that plumbing checks did not catch. A ROCm compiler and
rocBLAS bump is the same *class* of risk arriving through a different door: identical llama.cpp
sources, different generated kernels, different Tensile kernel selection for gfx1151. Everything
§6 establishes about preflight's blindness points the same way.

The gate's five clauses do not transfer literally — clauses 1 and 2 are about specific upstream
issues, and clause 5 (`make proof`) was recorded as "waived — the check does not exist". The
clauses that do transfer, and that I would require:

- **Clause 3 analogue** — `--direct-io` / `OLLAMA_IGPU_DIRECT_IO` re-validated on the new
  toolchain as a load-path integrity check, since it was validated under (c) on 7.2.4 specifically.
- **Clause 4 as written** — a vision A/B of **≥6 consecutive rows per build** on `qwen35moe` showing
  **0 degenerate** on the candidate, rollback boundary controlled. This is the clause that caught
  the original failure and it transfers unchanged.

---

## 8. What could not be verified without building

Everything in this section is **unknown**, not "probably fine". No image was pulled and none was
built (`/opt` has ~39 GB free; a ROCm base is ~26 GB decompressed).

1. **rocBLAS payload layout.** Whether TheRock's rocBLAS still installs
   `rocblas/library/TensileLibrary_lazy_gfx*.dat`. `discover/amd.go:119` and `Dockerfile:160` both
   depend on it. A layout change breaks GPU discovery **silently**.
2. **Whether the gfx90[06] prune on `Dockerfile:160` still matches anything**, and whether pruning
   is still the right size/granularity under TheRock's per-arch packaging.
3. **Whether `rocm/dev-ubuntu-*:7.14.1-full` actually contains what the build needs** — hipcc,
   HIP headers, rocBLAS *with gfx1151 kernels*, and a CMake-discoverable `/opt/rocm`. "full"
   replacing "complete" is an unexplained naming change and the contents were not inspected.
4. **Whether llama.cpp b10969's ggml-hip compiles clean against ROCm 7.14's LLVM/HIP**, including
   patch 903's `mmq.cu` hunk.
5. **Whether `AMDGPU_TARGETS` in `rocm_v7_2_linux` is still wholly valid on 7.14** — in particular
   whether `gfx908:xnack-` / `gfx90a:xnack±` survive, given `Dockerfile:160` already prunes
   gfx900/906 as unsupported.
6. **Build-time cost and image size** of any of the three migration paths.

---

## 9. Must be measured before this could ship

The current deployment is green on the vision suite, OCRBench and preflight. None of that
transfers across a toolchain change, because none of it is keyed on the toolchain (§6).

1. **Preflight** on the candidate, against a profile that *declares* the toolchain. Per ADR 0011
   this is a new `[profiles.…]` with its own measured `[expect.…]` blocks, generated and copied
   whole. **No existing row may be edited to make the candidate green** — ADR 0011 rule 4:

   > Copying another profile's numbers across to make a run green is the failure this file exists
   > to prevent.

2. **The gate's clause-4 vision A/B** — ≥6 consecutive rows per build on `qwen35moe`, 0 degenerate,
   rollback boundary controlled (§7).
3. **OCRBench**, against the current deployment as baseline. A toolchain change can move scores
   without moving any token count.
4. **The fine-text probe.** The 2026-09-19 promotion recorded a measured fine-text regression
   (`nemotron3` 9px −1, `qwen3.6` 7px −1, `qwen3.8` 9px +1, reproducible at N=5). That is the
   established baseline to compare against, not zero.
5. **gfx1151 GPU discovery**, explicitly — that `rocblasGFXTargets` still resolves `gfx1151` from
   the new payload (§8.1). Cheap to check, silent if it breaks.
6. **Portability floor.** If any path lands on Ubuntu, the glibc floor of the published amd64
   tarballs must be measured and the regression accepted deliberately (§3).

---

## 10. What this branch contains

This document, and nothing else. No `Dockerfile`, `CMakePresets.json`, `cmake/local.cmake`,
workflow, `expectations.toml` or test file was modified, because:

- there is no 7.14 value for `ARG ROCMVERSION` that resolves against `rocm/dev-almalinux-8`, so any
  Dockerfile edit would be a guess that cannot build (§2);
- the real change is a base-OS migration affecting all amd64 stages, whose target distro is a
  decision this investigation should inform rather than pre-empt (§3, §4);
- and a `rocm_v7_2` → `rocm_v7_14` rename touching 9 files is only coherent once the toolchain it
  names can actually be installed (§5).

Suggested next decision, in order: (a) confirm whether the goal is "the 7.14 line" or "a newer
ROCm on gfx1151" — if the latter, evaluate **10.0.0 on AlmaLinux 8 via `rhel8/` RPMs**, which is
the only path that preserves the base OS and the glibc floor (§4); (b) amend ADR 0011 to say
whether a toolchain-only change is a new profile, and add a `rocm_build` pin (§6); (c) confirm the
gate applies (§7). Only then is there a buildable branch to write.
