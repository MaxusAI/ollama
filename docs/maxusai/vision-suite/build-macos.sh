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
# 2026-08-17 — the first build stamped a bare "0.32.14" instead of
# "0.32.14-maxusai-<sha>", which no profile matches. A wrong stamp does not fail
# loudly; it makes preflight refuse to resolve a profile, or worse, resolve the
# wrong one.
set -eu

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"

OUT="${OUT:-/tmp/ollama-vs}"
JOBS="${JOBS:-8}"

# Base version = the newest upstream release tag reachable from HEAD, with any
# fork suffix stripped ("v0.32.14-dynres" -> "0.32.14"). Derived rather than
# hardcoded so a payload bump cannot leave a stale number in the stamp.
BASE="${BASE:-$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null \
  | sed 's/^v//; s/-.*$//')}"
[ -n "$BASE" ] || { echo "build-macos: could not derive a base version; set BASE=" >&2; exit 1; }
SHA="$(git rev-parse --short=8 HEAD)"
VERSION="${VERSION:-${BASE}-maxusai-${SHA}}"

# A dirty tree makes the sha a lie about what was built. Warn, do not block —
# building a work-in-progress for a quick check is legitimate.
if [ -n "$(git status --porcelain)" ]; then
  echo "build-macos: WARNING working tree is dirty; ${SHA} does not describe this binary" >&2
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
