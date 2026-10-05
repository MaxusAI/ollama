#!/usr/bin/env bash
# Build test-backend-ops and mmq-route for an AMD GPU, once per form of compat 903, with and without the debug hooks.
#
#   mmq-rocm-build.sh <llama.cpp checkout at b11081> <out dir>              forms of 903 on the fork's pin
#   mmq-rocm-build.sh --head <llama.cpp checkout at a #29953 head> <out dir> that head as fetched ("head")
#
# Prepares the checkout first (idempotent): mmq-tests.patch (as of bfc3fe487 for b11081, as of this file's tree for
# --head), and mmq-route.cpp as a test target. Binaries land in <out>/bin/<form>-<stock|debug>/ beside the mmq.cu
# diff they were built from. Static libraries, so each copy is self-contained; GGML_HIP_NO_VMM keeps its default
# (ON), so the pool is the legacy one that ships. The debug hooks' guard modes use HIP VMM directly.
# Environment: AMDGPU_TARGETS (gfx1151), JOBS (16), FORMS (none main widest amended).
set -eu
HEAD=0; [ "${1:-}" = --head ] && { HEAD=1; shift; }
SRC=$(cd "${1:?usage: $0 [--head] <llama.cpp checkout> <out dir>}" && pwd)
OUT=$(mkdir -p "${2:?usage: $0 [--head] <llama.cpp checkout> <out dir>}" && cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
B=$OUT/build
cd "$SRC"
if [ $HEAD = 0 ]; then
    [ "$(git rev-parse --short=9 HEAD)" = 161755f29 ] || { echo "$SRC is not at b11081 (161755f29)"; exit 1; }
    grep -q 'get_J_max(113) = 64' tests/test-backend-ops.cpp \
        || git -C "$HERE" show bfc3fe487:docs/maxusai/tasks/mmq-tests.patch | git apply
else
    grep -q 'get_J_max(113) = 64' tests/test-backend-ops.cpp || git apply "$HERE/mmq-tests.patch"
fi
cp "$HERE/mmq-route.cpp" tests/
grep -q mmq-route tests/CMakeLists.txt \
    || printf '\nadd_executable(mmq-route mmq-route.cpp)\ntarget_link_libraries(mmq-route PRIVATE ggml)\n' >> tests/CMakeLists.txt
cmake -S . -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx1151}" \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF > "$OUT/cmake.log" 2>&1
F=ggml/src/ggml-cuda/mmq.cu
forms=${FORMS:-none main widest amended}; [ $HEAD = 1 ] && forms=head
for v in $forms; do for e in stock debug; do
    d=$OUT/bin/$v-$e
    [ -x "$d/test-backend-ops" ] && [ -x "$d/mmq-route" ] && continue
    if [ $HEAD = 0 ]; then
        python3 "$HERE/mmq-variant-b11081.py" . "$v" $([ $e = debug ] && echo --debug) > /dev/null
    else
        git checkout -- "$F"
        [ $e = debug ] && python3 "$HERE/mmq-variant.py" . head --debug > /dev/null
    fi
    echo "building $v-$e ..."
    ninja -C "$B" -j "${JOBS:-16}" test-backend-ops mmq-route > "$OUT/build-$v-$e.log" 2>&1 \
        || { echo "build $v-$e failed, see $OUT/build-$v-$e.log"; exit 1; }
    mkdir -p "$d"; cp "$B/bin/test-backend-ops" "$B/bin/mmq-route" "$d/"
    git diff -- "$F" > "$d/mmq.cu.diff"
done; done
if [ $HEAD = 0 ]; then python3 "$HERE/mmq-variant-b11081.py" . none > /dev/null; else git checkout -- "$F"; fi
