#!/usr/bin/env bash
# Hand-routed MUL_MAT_ID cases (mmq-route.cpp) x forms of compat 903 x modes, one process per run.
#
#   mmq-rocm-route.sh <build out dir> [results dir, default <build out dir>/route]
#
# On RDNA3 the tile width follows the mean rows per expert, so test_mul_mat_id's uniform routing never leaves one
# column in the last tile of a wide launch; these cases build that routing by hand ("last1"), with uniform controls
# and the two worst cases mmq-rules-check.cu names for gfx1151 (chk_*).
# Environment: VARIANTS (none main widest amended), MODES (print guard_src1 guard_ids_dst), REPS (3), CASES.
set -u
BIN=$(cd "${1:?usage: $0 <build out dir> [results dir]}" && pwd)/bin
OUT=${2:-$1/route}
mkdir -p "$OUT/logs"; ulimit -c 0
VARIANTS=${VARIANTS:-none main widest amended}
MODES=${MODES:-print guard_src1 guard_ids_dst}
REPS=${REPS:-3}
# name -> mmq-route args: type n_exp n_used n_tokens m k bcast route
declare -A C=(
    [q2k_e8_last1_b0]='q2_K 8 7 74 256 1024 0 last1'
    [q2k_e8_last1_b1]='q2_K 8 7 74 256 1024 1 last1'
    [q2k_e8_unif_b0]='q2_K 8 7 74 256 1024 0 uniform'
    [q2k_e128_last1_b1]='q2_K 128 16 600 256 1024 1 last1'
    [q2k_e128_unif_b1]='q2_K 128 16 600 256 1024 1 uniform'
    [q4k_e128_u6_n5_b1]='q4_K 128 6 5 512 2048 1 uniform'
    [chk_65_8_8]='q2_K 8 8 65 256 1024 0 uniform'
    [chk_513_1_8]='q2_K 8 1 513 256 1024 0 uniform'
)
CASES=${CASES:-q2k_e8_last1_b0 q2k_e8_last1_b1 q2k_e8_unif_b0 q2k_e128_last1_b1 q2k_e128_unif_b1 q4k_e128_u6_n5_b1 chk_65_8_8 chk_513_1_8}
declare -A M_ENV=( [print]=off [guard_src1]=guard:src1 [guard_ids_dst]=guard:ids_dst [guard]=guard )
TSV=$OUT/results.tsv
[ -s "$TSV" ] || printf 'case\tvariant\tmode\trep\tok\tillegal_access\texit\troute\tmmq_id\n' > "$TSV"
for c in $CASES; do for v in $VARIANTS; do for mode in $MODES; do
    for r in $(seq 1 "$REPS"); do
        log=$OUT/logs/${c}_${v}_${mode}_r$r.log
        grep -q '^RESULT' "$log" 2>/dev/null && continue
        env MMQ_DEBUG_ALLOC="${M_ENV[$mode]}" MMQ_DEBUG_PRINT=1 timeout 600 "$BIN/$v-debug/mmq-route" ${C[$c]} > "$log" 2>&1
        rc=$?
        ok=$(grep -q -- '-> OK' "$log" && echo 1 || echo 0)
        ill=$(grep -ciE 'illegal memory access|memory access fault' "$log")
        route=$(grep -m1 -oE 'rows=.*' "$log")
        id=$(grep -m1 -E '^MMQ_(ID|IDS|DENSE)' "$log" | sed -E 's/ +/ /g')
        printf 'RESULT\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$ok" "$ill" "$rc" >> "$log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$c" "$v" "$mode" "$r" "$ok" "$ill" "$rc" "$route" "$id" >> "$TSV"
        printf '%-20s %-8s %-14s r%s ok=%s illegal=%s rc=%s\n' "$c" "$v" "$mode" "$r" "$ok" "$ill" "$rc"
    done
done; done; done
