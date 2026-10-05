#!/usr/bin/env bash
# The HIP counterpart of mmq-29953-guard-verify.sh (MaxusAI/ollama#455): the same builds of mmq.cu, with hipcc for
# gfx1151, on llama.cpp#29953's head (3070d927f) with mmq-29953-y-tile-guard.patch, each scored by WHY it failed:
#   1.  the head as is, unpatched                         -> must compile (baseline time)
#   2.  the head with the patch                           -> must compile
#   3.  helper shrunk to J blocks                         -> must FAIL by the guard, naming every table
#   4.  helper rounded to one warp's ints                 -> must FAIL by the guard, naming every table
#   5a. rdna3_5's first 256-thread config at 192 threads  -> must COMPILE. That config has J = 64, and 64*36 ints is
#       exactly 12*192, so the bit mask in GGML_PAD happens to round correctly and there is nothing to catch. No
#       rdna3_5 config can be made short with 192 threads: only J = 8, 56, 72 and 120 can, and rdna3_5 uses none.
#   5b. the same config at 160 threads                    -> must FAIL by the guard, naming ONLY rdna3_5: GGML_PAD's
#       bit mask gives 9216 B, and the load loop reads 15*160 ints = 9600 B
#   6.  the guard compiled out                            -> must compile (timing); its object code is compared with
#       build 2's, to show the guard adds none
# clang stops at 20 errors by default, which would name only the first tables; -ferror-limit=0 lifts it.
#
# Under clang, the patch as published reports ONE failing type per table, and then a knock-on error from the outer
# static_assert(ggml_cuda_mmq_y_tile_all_<table>(...) > 0) at the same line: "static assertion expression is not an
# integral constant expression" -- the same words clang uses when it hits -fconstexpr-steps. The scoring below counts
# the knock-on as not the guard's, as the nvcc script does. VARIANT=instantiate replaces the function template and that
# outer static_assert with an explicitly instantiated class that inherits every per-type guard, so nothing outside
# the per-type assertions is constant-evaluated; clang then reports every failing (table, type) once, and nothing else.
#
# Usage: mmq-29953-guard-verify-hip.sh <llama.cpp clone that has 3070d927f> [cmake|hipcc]
#   cmake (default): the compile command CMake generates for mmq.cu (GGML_HIP=ON, gfx1151, Release), run directly,
#                    with no ccache -- the way compat 903's guard was measured for #451
#   hipcc:           hipcc -std=c++17 -DGGML_USE_HIP --offload-arch=gfx1151 -I. -Iggml/include -Iggml/src
#                    -Iggml/src/ggml-cuda, the minimal command, matching the nvcc script's
# REPEATS=N also times builds 2 and 6, interleaved, N times each. VARIANT=instantiate applies the variant above on top
# of the patch. Results go to ./verify-29953-guard-hip-<mode>[-<variant>].
set -u
[ $# -ge 1 ] || { sed -n '2,34p' "$0"; exit 2; }
LC=$(realpath "$1"); MODE=${2:-cmake}
HERE=$(cd "$(dirname "$0")" && pwd)
PATCH=${PATCH:-$HERE/mmq-29953-y-tile-guard.patch}
ROCM=${ROCM_PATH:-/opt/rocm}
VARIANT=${VARIANT:-}
OUT=${OUT:-$PWD/verify-29953-guard-hip-$MODE${VARIANT:+-$VARIANT}}
REPEATS=${REPEATS:-0}
case "$VARIANT" in ''|instantiate) ;; *) echo "VARIANT must be empty or 'instantiate'"; exit 2;; esac
HEAD29953=3070d927f6c172242d56a2c1fb897e838e7b2632
WT=$OUT/wt
rm -rf "$OUT"; mkdir -p "$OUT"
echo "host load at start (1/5/15 min): $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) cores"
echo "ROCm $(cat "$ROCM/.info/version" 2>/dev/null), $("$ROCM/bin/hipcc" --version 2>&1 | grep -m1 'HIP version'), $("$ROCM/lib/llvm/bin/clang++" --version | head -1)"

git -C "$LC" worktree prune
git -C "$LC" worktree add --quiet --detach "$WT" "$HEAD29953" || exit 1
trap 'git -C "$LC" worktree remove --force "$WT" 2>/dev/null' EXIT
cd "$WT" || exit 1
CU=ggml/src/ggml-cuda/mmq.cu; CUH=ggml/src/ggml-cuda/mmq.cuh; R35=ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh

if [ "$MODE" = cmake ]; then
    cmake -S "$WT" -B "$OUT/build" -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CCACHE=OFF -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
        > "$OUT/cmake.log" 2>&1 || { echo "FAIL: cmake configure, see $OUT/cmake.log"; exit 1; }
    # CMake's own command for mmq.cu, minus the dependency-file and output flags, plus -ferror-limit=0
    mapfile -t CMD < <(python3 - "$OUT/build/compile_commands.json" "$WT/$CU" "$OUT/mmq.o" <<'PY'
import json, os, shlex, sys
db, src, out = sys.argv[1:4]
e = next(e for e in json.load(open(db)) if os.path.realpath(e["file"]) == os.path.realpath(src))
args, keep, skip = e.get("arguments") or shlex.split(e["command"]), [], False
for a in args:
    if skip:
        skip = False
    elif a in ("-o", "-c", "-MT", "-MF"):
        skip = True
    elif a not in ("-MD", "-MMD"):
        keep.append(a)
print("\n".join([e["directory"]] + keep + ["-ferror-limit=0", "-c", src, "-o", out]))
PY
)
    RUNDIR=${CMD[0]}; CMD=("${CMD[@]:1}")
    [ "${#CMD[@]}" -gt 4 ] || { echo "FAIL: no compile command for $CU"; exit 1; }
else
    RUNDIR=$WT
    CMD=("$ROCM/bin/hipcc" -std=c++17 -DGGML_USE_HIP --offload-arch=gfx1151 -I. -Iggml/include -Iggml/src
         -Iggml/src/ggml-cuda -ferror-limit=0 -c "$WT/$CU" -o "$OUT/mmq.o")
fi
printf '%s\n' "${CMD[@]}" > "$OUT/command.txt"

secs() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.2f", b-a}'; }
compile() { # <log>: one compile of mmq.cu; prints the seconds, returns the compiler's exit code
    local t0 t1 rc
    t0=$(date +%s.%N); (cd "$RUNDIR" && "${CMD[@]}") > "$1" 2>&1; rc=$?; t1=$(date +%s.%N)
    secs "$t0" "$t1"; return $rc
}
GUARD_MSG='MMQ: the y tile helper is smaller than the y load'
build() { # <label> <expect ok|fail> [expected table list, space-separated, sorted]
    local label=$1 expect=$2 want_tables=${3:-} log="$OUT/$1.log" t rc
    t=$(compile "$log"); rc=$?
    local got=ok; [ $rc -ne 0 ] && got=fail
    # count errors, not lines; clang prints each on one line ("file:line:col: error: ..."), notes separately
    local nerr nguard tables
    nerr=$(grep -c 'error: ' "$log")
    nguard=$(grep -c "error: static assertion failed.*$GUARD_MSG" "$log")
    tables=$(grep "error: static assertion failed.*$GUARD_MSG" "$log" | grep -oE 'in the [a-z0-9_]+ table' \
             | sed 's/in the //;s/ table//' | sort -u | tr '\n' ' ' | sed 's/ $//')
    local verdict=UNEXPECTED
    if [ "$got" = "$expect" ]; then
        if [ "$expect" = ok ]; then verdict=as-expected
        elif [ "$nguard" -gt 0 ] && [ "$nerr" -eq "$nguard" ] && [ "$tables" = "$want_tables" ]; then verdict=as-expected
        else verdict=WRONG-REASON
        fi
    fi
    printf '%-36s rc=%-3s %-4s want=%-4s %-13s %6ss\n' "$label" "$rc" "$got" "$expect" "$verdict" "$t"
    [ "$got" = fail ] && printf '    errors=%s, from the guard=%s, tables named: %s\n' "$nerr" "$nguard" "${tables:-none}"
    [ "$verdict" = as-expected ]
}
swap() { # <file> <old> <new>: exact one-occurrence string swap, so a mutation cannot fail for a typo of mine
    python3 - "$1" "$2" "$3" <<'PY' || exit 1
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
if s.count(old) != 1:
    sys.exit("mutation anchor found %d times in %s, expected once: %r" % (s.count(old), p, old))
open(p, "w").write(s.replace(old, new))
PY
}

ALL="ampere blackwell cdna gcn pascal_dp4a pascal_older rdna2 rdna3 rdna3_5 rdna4"
ok=1
build "1-head-unpatched" ok || ok=0

git apply "$PATCH" || { echo "FAIL: the patch does not apply to $HEAD29953"; exit 1; }
echo "patch applies to a pristine $HEAD29953"
if [ "$VARIANT" = instantiate ]; then
    python3 - "$CU" <<'PY' || exit 1
import sys
p = sys.argv[1]; s = open(p).read()
head = "    template <int... ts> static constexpr size_t ggml_cuda_mmq_y_tile_all_##TABLE("
tail = 'static_assert(ggml_cuda_mmq_y_tile_all_##TABLE(std::make_integer_sequence<int, GGML_TYPE_COUNT>()) > 0, "");'
if s.count(head) != 1 or s.count(tail) != 1:
    sys.exit("variant anchors found %d and %d times, expected once each" % (s.count(head), s.count(tail)))
a = s.index(head); b = s.index(tail) + len(tail)
new = [
    "    template <typename> struct ggml_cuda_mmq_y_tile_guards_##TABLE;",
    "    template <int... ts> struct ggml_cuda_mmq_y_tile_guards_##TABLE<std::integer_sequence<int, ts...>>",
    "        : ggml_cuda_mmq_y_tile_guard_##TABLE<ts>... {};",
    "    template struct ggml_cuda_mmq_y_tile_guards_##TABLE<std::make_integer_sequence<int, GGML_TYPE_COUNT>>;",
]
new = "\n".join([l.ljust(120) + "\\" for l in new[:-1]] + new[-1:])
open(p, "w").write(s[:a] + new + s[b:])
PY
    echo "variant 'instantiate' applied: the guard's tail is now"
    grep -A3 -F '    template <typename> struct ggml_cuda_mmq_y_tile_guards_##TABLE;' "$CU" | sed 's/ *\\$//; s/^/    /'
fi
cp "$CU" "$OUT/mmq.cu.patched"; cp "$CUH" "$OUT/mmq.cuh.patched"; cp "$R35" "$OUT/rdna3-5.orig"
build "2-head-with-guard" ok || ok=0
cp "$OUT/mmq.o" "$OUT/mmq-2.o" 2>/dev/null

HELPER='    return GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));'
swap "$CUH" "$HELPER" '    return config.J*sizeof(block_q8_1_mmq);'
build "3-helper-shrunk-to-J-blocks" fail "$ALL" || ok=0
cp "$OUT/mmq.cuh.patched" "$CUH"

swap "$CUH" "$HELPER" '    return GGML_PAD(config.J*sizeof(block_q8_1_mmq), 32*sizeof(int));'
build "4-helper-rounded-to-one-warp" fail "$ALL" || ok=0
cp "$OUT/mmq.cuh.patched" "$CUH"

# the whole first CASE line with 256 threads: the type alone repeats once per J, so it is not a unique anchor
first=$(grep -m1 -E '^ *CASE\(GGML_TYPE_[A-Z0-9_]+, *256,' "$R35")
swap "$R35" "$first" "${first/256,/192,}"
echo "mutation 5a rewrote: $first -> ${first/256,/192,}"
build "5a-rdna3_5-config-at-192-threads" ok || ok=0
cp "$OUT/rdna3-5.orig" "$R35"

swap "$R35" "$first" "${first/256,/160,}"
echo "mutation 5b rewrote: $first -> ${first/256,/160,}"
build "5b-rdna3_5-config-at-160-threads" fail "rdna3_5" || ok=0
cp "$OUT/rdna3-5.orig" "$R35"

sed 's/^#    define GGML_CUDA_MMQ_Y_TILE_GUARD_ON$/\/\/ guard disabled for timing/; s/^#        define GGML_CUDA_MMQ_Y_TILE_GUARD_ON$/\/\/ guard disabled for timing/' \
    "$OUT/mmq.cu.patched" > "$OUT/mmq.cu.unguarded"
cmp -s "$OUT/mmq.cu.patched" "$OUT/mmq.cu.unguarded" && { echo "FAIL: the guard's defines were not found"; exit 1; }
cp "$OUT/mmq.cu.unguarded" "$CU"
build "6-guard-compiled-out" ok || ok=0
cp "$OUT/mmq.o" "$OUT/mmq-6.o" 2>/dev/null
cp "$OUT/mmq.cu.patched" "$CU"

# the guard is meant to cost compile time only: builds 2 and 6 must emit the same object. Both are compiled from the
# same path to the same output name, so even __hip_cuid_* (a hash of the file path and the command line) matches;
# if the bytes differ, the host disassembly and the section sizes say whether any of it is code
if [ -f "$OUT/mmq-2.o" ] && [ -f "$OUT/mmq-6.o" ]; then
    if cmp -s "$OUT/mmq-2.o" "$OUT/mmq-6.o"; then
        echo "object code, build 2 vs build 6 (guard on vs off): byte-identical, $(stat -c %s "$OUT/mmq-2.o") B"
    else
        dis() { objdump -d --no-show-raw-insn "$1" | tail -n +4; }
        sec() { size -A "$1" | tail -n +3; }
        same=identical; cmp -s <(dis "$OUT/mmq-2.o") <(dis "$OUT/mmq-6.o") || same=DIFFERENT
        sizes=identical; cmp -s <(sec "$OUT/mmq-2.o") <(sec "$OUT/mmq-6.o") || sizes=DIFFERENT
        echo "object code, build 2 vs build 6 (guard on vs off): bytes differ; host disassembly $same, section sizes $sizes"
        [ "$same" = identical ] && [ "$sizes" = identical ] || ok=0
    fi
fi

if [ "$REPEATS" -gt 0 ]; then
    with=(); without=()
    for r in $(seq "$REPEATS"); do
        cp "$OUT/mmq.cu.patched" "$CU";   with+=("$(compile "$OUT/time-with.log")")
        cp "$OUT/mmq.cu.unguarded" "$CU"; without+=("$(compile "$OUT/time-without.log")")
    done
    cp "$OUT/mmq.cu.patched" "$CU"
    med() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
    echo "timing, $REPEATS interleaved repeats: with the guard ${with[*]} s (median $(med "${with[@]}")),"\
         "without ${without[*]} s (median $(med "${without[@]}"))"
fi

echo "VERIFY29953GUARD-HIP COMPLETE $(date -u +%FT%TZ) mode=$MODE variant=${VARIANT:-none} all_as_expected=$ok"
