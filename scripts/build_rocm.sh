#!/bin/sh
# Build the fork's ROCm image from Dockerfile.rocm (ADR 0042): every stage on
# rocm/dev-ubuntu-24.04, never upstream's Dockerfile and never rocm/dev-almalinux-8.
#
#   ROCM_TOOLCHAIN   rocm7 (default): rocm/dev-ubuntu-24.04:7.2.4-complete, preset rocm_v7_2
#                    rocm10:          rocm/dev-ubuntu-24.04:10.0.0-full,   preset rocm_v10_0 (experimental, ADR 0040)
#   AMDGPU_TARGETS   optional, e.g. gfx1151 -- a single-arch dev or probe build
#   CONTEXT          source tree to build (default: this checkout), so another worktree
#                    can be built with this recipe without copying it there
#   TAG              image tag (default: maxusai-ollama:$VERSION-<toolchain>[-<targets>])
#   PARALLEL         build jobs (default: nproc - 1)
#
# The version is stamped by scripts/env.sh from CONTEXT's own git history, as for every
# other build (ADR 0032); set VERSION to override it.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
CONTEXT=$(cd "${CONTEXT:-$here}" && pwd)
ROCM_TOOLCHAIN=${ROCM_TOOLCHAIN:-rocm7}
case "$ROCM_TOOLCHAIN" in
  rocm7)  ROCM_TAG=7.2.4-complete; ROCM_VARIANT=rocm_v7_2 ;;
  rocm10) ROCM_TAG=10.0.0-full;    ROCM_VARIANT=rocm_v10_0 ;;
  *) echo "build_rocm.sh: ROCM_TOOLCHAIN must be rocm7 or rocm10, not '$ROCM_TOOLCHAIN'" >&2; exit 2 ;;
esac
ROCM_IMAGE=rocm/dev-ubuntu-24.04

if ! grep -q "\"${ROCM_VARIANT}_linux\"" "$CONTEXT/llama/server/CMakePresets.json"; then
  echo "build_rocm.sh: $CONTEXT has no ${ROCM_VARIANT}_linux preset." >&2
  echo "  The rocm_v10_0 presets arrived with ADR 0042; a tree older than that cannot build rocm10." >&2
  exit 2
fi

cd "$CONTEXT"
export PLATFORM=linux/amd64
. "$here/scripts/env.sh"
targets_suffix=""
if [ -n "${AMDGPU_TARGETS:-}" ]; then
  targets_suffix="-$(printf '%s' "$AMDGPU_TARGETS" | tr ';:,' '---')"
fi
TAG=${TAG:-maxusai-ollama:${VERSION}-${ROCM_TOOLCHAIN}${targets_suffix}}
PARALLEL=${PARALLEL:-$(( $(nproc) - 1 ))}

echo "build_rocm.sh: $ROCM_IMAGE:$ROCM_TAG ($ROCM_VARIANT) -> $TAG"
echo "  context: $CONTEXT @ $(git -C "$CONTEXT" rev-parse --short=9 HEAD)"
echo "  targets: ${AMDGPU_TARGETS:-preset default}  jobs: $PARALLEL"

# shellcheck disable=SC2086 # OLLAMA_COMMON_BUILD_ARGS is a word list by design (env.sh)
docker buildx build --load --progress=plain \
    --platform=linux/amd64 \
    ${OLLAMA_COMMON_BUILD_ARGS} \
    --build-arg "ROCM_IMAGE=$ROCM_IMAGE" \
    --build-arg "ROCM_TAG=$ROCM_TAG" \
    --build-arg "ROCM_VARIANT=$ROCM_VARIANT" \
    --build-arg "PARALLEL=$PARALLEL" \
    ${AMDGPU_TARGETS:+--build-arg "AMDGPU_TARGETS=$AMDGPU_TARGETS"} \
    -f "$here/Dockerfile.rocm" \
    -t "$TAG" \
    "$CONTEXT"

docker run --rm --entrypoint sh "$TAG" -c \
  "/bin/ollama -v 2>&1 | tail -1; cat /usr/lib/ollama/$ROCM_VARIANT/ROCM_VERSION /usr/lib/ollama/$ROCM_VARIANT/ROCM_IMAGE"
