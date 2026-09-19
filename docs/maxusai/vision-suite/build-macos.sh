#!/bin/sh
# Build the native macOS fork binary used for benchmarking, with a version stamp
# that identifies it as a fork artifact.
#
#   sh docs/maxusai/vision-suite/build-macos.sh              # -> /tmp/ollama-vs
#   OUT=./ollama sh docs/maxusai/vision-suite/build-macos.sh
#
# WHY THIS IS A SCRIPT AND NOT PROSE. The binary is the provenance for every
# measurement in this repository, and preflight gates on the version string it
# reports (ADR 0011: expectations are keyed on (platform, version)). Assembling
# the ldflags by hand from spec/apple-silicon-build.md got it wrong on
# 2026-08-17 — the first build stamped a bare "0.32.14", which no profile
# matches. A wrong stamp does not fail loudly; it makes preflight refuse to
# resolve a profile, or worse, resolve the wrong one.
#
#   STAMP_ONLY=1 sh docs/maxusai/vision-suite/build-macos.sh   # print the stamp, build nothing
set -eu

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

OUT="${OUT:-/tmp/ollama-vs}"
JOBS="${JOBS:-8}"

# The stamp is scripts/env.sh's — the same `git describe --tags --first-parent`
# every other build uses (ADR 0032): <release>-dynres[.N]-<n>-g<sha>[-dirty].
#
# It used to be this script's own "<base>-maxusai-<sha>", which diverged from the
# fold's identity twice over: a different shape, and, because the base was the
# newest release tag at BUILD time, the wrong release whenever a build ran before
# its fold tag was cut. The v0.34.1 Metal build (12:00:24, 2026-09-18) preceded
# `v0.34.1-dynres` (12:49:45) and stamped 0.34.0-maxusai-8a7ba949 for the commit
# the CUDA image stamps 0.34.1-dynres-0-g8a7ba94 (ADR 0032, 2026-09-19 amendment).
#
# CUT THE FOLD TAG BEFORE BUILDING THE FOLD. A build made first describes from the
# previous tag, and the mlx-metal profile refuses that stamp, so the mistake now
# stops at preflight instead of shipping.
#
# env.sh is sourced in a subshell so only VERSION crosses over: its release
# GOFLAGS (-s -w, server.mode=release) would change what this script builds.
VERSION="${VERSION:-$(. ./scripts/env.sh >/dev/null 2>&1; printf %s "$VERSION")}"
[ -n "$VERSION" ] || { echo "build-macos: could not derive a version; set VERSION=" >&2; exit 1; }

# Before anything writes to stdout or touches the build tree.
if [ -n "${STAMP_ONLY:-}" ]; then
  printf '%s\n' "$VERSION"
  exit 0
fi

# A dirty tree describes no commit exactly. The stamp now says so itself (the
# -dirty suffix, which no profile admits); warn as well, do not block — building
# a work-in-progress for a quick check is legitimate.
if [ -n "$(git status --porcelain)" ]; then
  echo "build-macos: WARNING working tree is dirty; the stamp ${VERSION} names no commit exactly" >&2
fi

echo "build-macos: version ${VERSION}"
echo "build-macos: payload  $(cat LLAMA_CPP_VERSION)  mlx $(cut -c1-12 MLX_VERSION)"

# The native payload. CLEAN_DEPS=1 removes the vendored llama.cpp checkout
# first: the compat patches are applied to it as working-tree edits, and when
# LLAMA_CPP_VERSION moves, CMake's stash/fetch/unstash cycle fails with
# "Failed to unstash changes ... resolve the conflicts manually". Clearing the
# checkout lets the applier re-apply all six patches against the new tag, which
# is the documented recovery (llama/compat/README.md).
if [ "${CLEAN_DEPS:-0}" = "1" ]; then
  echo "build-macos: clearing vendored llama.cpp checkout"
  rm -rf build/_deps/llama_cpp-src build/_deps/llama_cpp-subbuild \
         build/ollama-llama-cpp-source-prefix
fi

cmake -B build .
cmake --build build --parallel "$JOBS"

go build -trimpath \
  -ldflags="-X=github.com/ollama/ollama/version.Version=${VERSION}" \
  -o "$OUT" .

echo "build-macos: wrote ${OUT}"
OLLAMA_HOST=127.0.0.1:1 "$OUT" --version 2>&1 | sed -n 's/^Warning: client version is /build-macos: stamped /p'
