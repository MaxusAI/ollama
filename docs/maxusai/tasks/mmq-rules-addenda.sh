#!/usr/bin/env bash
# The two addenda to the main matrix, run back to back. No pgrep waiting: the earlier chained versions looped on
# `pgrep -f "mmq-rules-gpu.sh llama.cpp-e2e"`, which matched the shell that had just created them (the pattern was
# in its own argv), so neither ever started. Nothing else is running now, so just run them in order.
#
#   A. the combined guard mode (src1 and ids_dst guarded at once) for the five cases that carry the argument
#   B. j100_b1, j100_b0 and ids16 under guard:src1, ten reps, with the amended rule as a control in the same
#      harness: #29953's shortfall there is one 144-byte block, which is marginal, and 1 of 2 is not a result.
set -u
cd "$(dirname "$0")"
LOG=run-addenda.log
: > "$LOG"

echo "=== A: combined guard, 5 cases x 7 rules ===" >> "$LOG"
VARIANTS="ne11 ne12 p27044 successor s448p p29953 p29953fix" \
  CASES="ids16 ids64 e120_b1 one113 j100_b0" MODES="guard" REPS=2 JOBS=8 \
  ./mmq-rules-gpu.sh llama.cpp-e2e "$PWD/mmq-rules-results" >> "$LOG" 2>&1

echo "=== B: guard:src1 repeats, 3 cases x 3 rules x 10 ===" >> "$LOG"
VARIANTS="p29953 p29953fix ne12" CASES="j100_b1 j100_b0 ids16" MODES="guard_src1" REPS=10 JOBS=8 \
  ./mmq-rules-gpu.sh llama.cpp-e2e "$PWD/mmq-rules-results" >> "$LOG" 2>&1

./mmq-rules-table.py mmq-rules-results/results.tsv > mmq-rules-results/matrix.md
{
  echo "ADDENDA COMPLETE $(date -u +%FT%TZ) rows=$(wc -l < mmq-rules-results/results.tsv)"
  echo "guard:src1 pass rates (the marginal cell and its control):"
  awk -F'\t' '$3=="guard_src1" && $1 ~ /^(j100_b0|j100_b1|ids16)$/ {k=$1" "$2; n[k]++; if ($5=="1") p[k]++}
              END {for (k in n) printf "  %-24s %2d/%-2d passed\n", k, p[k]+0, n[k]}' \
      mmq-rules-results/results.tsv | sort
} >> "$LOG"
