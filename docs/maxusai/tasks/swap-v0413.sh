#!/bin/bash
# Tag v0.40.13-dynres (85d636818, #501's merge) as a Go-only swap onto sync-0.40.13-fold, the image that passed gates
# 3-5 on CUDA. The maintainer's word of 2026-11-04: "switch to the Go-only swap" (from "build the tagged v0.40.13
# image and deploy it to CUDA"). 81dda7d30..tag changes Go source and docs only (checked below), so a full build would
# differ from this image in /usr/bin/ollama alone. Derived from swap-0403-main.sh: native-input refusal, publish-go on
# bigdisk, Dockerfile.swap, verify stamp + binary sha + every payload file hash-identical to BASE.
set -u
S=/mnt/8TB_SN850X_RAID1_BTRFS/claude-scratch; REPO=/opt/github/MaxusAI/ollama
SRCREF=HEAD; TAGBASE=81dda7d30
W=$S/wt-v0413tag; OUT=$S/gobin-v0413
BASE=maxusai/ollama:sync-0.40.13-fold; TAG=maxusai/ollama:v0.40.13-dynres
die() { echo "### SWAP-V0413 FAILED $(date '+%F %T'): $*"; exit 1; }
echo "### SWAP-V0413 start $(date '+%F %T')"
docker image inspect "$BASE" >/dev/null 2>&1 || die "missing $BASE"
docker image inspect "$TAG" >/dev/null 2>&1 && die "$TAG already exists"
native=$(git -C "$W" diff --name-only "$TAGBASE" HEAD -- cmake CMakeLists.txt CMakePresets.json llama mlx MLX_C_VERSION \
  MLX_VERSION LLAMA_CPP_VERSION mlxrunner/xgrammar/native go.mod go.sum Dockerfile ':!llama/compat/README.md')
[ -z "$native" ] || die "native inputs changed since $TAGBASE: $(echo $native)"
echo "### native inputs unchanged since $TAGBASE"
cd "$W" || die "cd"
. scripts/env.sh >/dev/null
echo "### VERSION=$VERSION"
rm -rf "$OUT"; mkdir -p "$OUT"
docker buildx build --builder bigdisk --platform=linux/amd64 ${OLLAMA_COMMON_BUILD_ARGS} --target publish-go \
  --output type=local,dest="$OUT" . > "$OUT.build.log" 2>&1 || die "publish-go failed (see $OUT.build.log)"
[ -x "$OUT/bin/ollama" ] || die "no binary"
printf 'FROM %s\nCOPY bin/ollama /usr/bin/ollama\nCOPY bin/ollama /bin/ollama\n' "$BASE" > "$OUT/Dockerfile.swap"
docker build -q -f "$OUT/Dockerfile.swap" -t "$TAG" "$OUT" >/dev/null || die "swap build failed"
want=$(sha256sum "$OUT/bin/ollama" | cut -c1-64)
for p in /usr/bin/ollama /bin/ollama; do
  got=$(docker run --rm --entrypoint sha256sum "$TAG" "$p" | cut -c1-64)
  [ "$got" = "$want" ] || die "$p sha $got != built $want"
done
stamp=$(docker run --rm --entrypoint /bin/ollama "$TAG" --version 2>&1 | tail -1 | sed 's/.* //')
[ "$stamp" = "$VERSION" ] || die "stamp $stamp != $VERSION"
hp() { docker run --rm --entrypoint sh "$1" -c 'cd /usr/lib/ollama && find . -type f -print0 | sort -z | xargs -0 sha256sum' ; }
hp "$BASE" > "$OUT/payload-base.sha"; hp "$TAG" > "$OUT/payload-swap.sha"
cmp -s "$OUT/payload-base.sha" "$OUT/payload-swap.sha" || die "payload differs from BASE"
echo "### payload: $(wc -l < "$OUT/payload-swap.sha") files hash-identical to $BASE"
echo "### SWAP-V0413 DONE $(date '+%F %T'): $TAG stamps $stamp, binary $want"
