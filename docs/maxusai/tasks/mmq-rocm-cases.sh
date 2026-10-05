#!/usr/bin/env bash
# ROCm port of mmq-rules-gpu.sh: the twelve mmq-tests.patch cases x forms of compat 903 x modes, one process per run.
#
#   mmq-rocm-cases.sh <build out dir> [results dir, default <build out dir>/cases]
#
# Binaries come from mmq-rocm-build.sh. No memcheck or exact modes: there is no compute-sanitizer for ROCm here, and
# device ASan needs xnack, which gfx1151 lacks. The guard modes use mmq-debug-alloc.cuh's allocator on HIP VMM.
# Environment: VARIANTS (none main widest amended), MODES (stock guard_ids_dst guard_src1), REPS (3), CASES (the
# twelve; dense321 is #448's sm_75 MUL_MAT case, in mmq-tests.patch only from 1ecdb175c, so --head builds only).
set -u
BIN=$(cd "${1:?usage: $0 <build out dir> [results dir]}" && pwd)/bin
OUT=${2:-$1/cases}
mkdir -p "$OUT/logs"
ulimit -c 0
VARIANTS=${VARIANTS:-none main widest amended}
MODES=${MODES:-stock guard_ids_dst guard_src1}
REPS=${REPS:-3}
CASES_ALL="p29847_b0 p29847_b1 j100_b0 j100_b1 e120_b0 e120_b1 one113 orig2040 fb65 t100 ids16 ids64"
CASES=${CASES:-$CASES_ALL}
declare -A P=(
    [p29847_b0]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=0,m=640,n=508,k=2560,'
    [p29847_b1]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=508,k=2560,'
    [j100_b0]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=0,m=640,n=100,k=2560,'
    [j100_b1]='type_a=q4_0,type_b=f32,n_mats=512,n_used=10,b=1,m=640,n=100,k=2560,'
    [e120_b0]='type_a=q4_K,type_b=f32,n_mats=256,n_used=10,b=0,m=576,n=120,k=1536,'
    [e120_b1]='type_a=q4_K,type_b=f32,n_mats=256,n_used=10,b=1,m=576,n=120,k=1536,'
    [one113]='type_a=q4_K,type_b=f32,n_mats=64,n_used=1,b=0,m=576,n=113,k=16384,'
    [orig2040]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=512,n=2040,k=2048,'
    [fb65]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=576,n=65,k=2048,'
    [t100]='type_a=q4_K,type_b=f32,n_mats=256,n_used=8,b=1,m=512,n=100,k=2048,'
    [ids16]='type_a=q4_0,type_b=f32,n_mats=256,n_used=16,b=0,m=640,n=16,k=2560,'
    [ids64]='type_a=q4_0,type_b=f32,n_mats=256,n_used=64,b=0,m=640,n=100,k=2560,'
    [dense321]='type_a=q2_K,type_b=f32,m=512,n=321,k=1024,'
)
declare -A OP=( [dense321]=MUL_MAT )
declare -A M_BIN=( [stock]=stock [guard]=debug [guard_ids_dst]=debug [guard_src1]=debug [print]=debug )
declare -A M_ENV=( [stock]=off [guard]=guard [guard_ids_dst]=guard:ids_dst [guard_src1]=guard:src1 [print]=off )
TSV=$OUT/results.tsv
[ -s "$TSV" ] || printf 'case\tvariant\tmode\trep\tpassed\tillegal_access\texit\tmmq_id\n' > "$TSV"
for c in $CASES; do for v in $VARIANTS; do for mode in $MODES; do
    bin=$BIN/$v-${M_BIN[$mode]}/test-backend-ops
    for r in $(seq 1 "$REPS"); do
        log=$OUT/logs/${c}_${v}_${mode}_r$r.log
        grep -q '^RESULT' "$log" 2>/dev/null && continue
        env MMQ_DEBUG_ALLOC="${M_ENV[$mode]}" MMQ_DEBUG_PRINT=1 timeout 900 "$bin" test -o "${OP[$c]:-MUL_MAT_ID}" -b ROCm0 -p "${P[$c]}" > "$log" 2>&1
        rc=$?
        pass=$(grep -qE '^ *1/1 tests passed' "$log" && echo 1 || echo 0)
        ill=$(grep -ciE 'illegal memory access|memory access fault' "$log")
        id=$(grep -m1 -E '^MMQ_(ID|IDS|DENSE)' "$log" | sed -E 's/ +/ /g')
        printf 'RESULT\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill" "$rc" >> "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill" "$rc" "$id" >> "$TSV"
        printf '%-10s %-8s %-14s r%s pass=%s illegal=%s rc=%s\n' "$c" "$v" "$mode" "$r" "$pass" "$ill" "$rc"
    done
done; done; done
