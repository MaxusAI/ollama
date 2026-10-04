#!/usr/bin/env python3
"""Render mmq-rules-gpu.sh's results.tsv as one markdown table per mode (ADR 0012: tables come from the generator).

    mmq-rules-table.py results.tsv [results.tsv ...]

A cell reads "p/n" for the runs that passed of the runs made. A mode run under compute-sanitizer also carries its
error count, "E=<n>"; "abort" means the process died instead of finishing the case.
"""
import collections
import csv
import sys

MODE_LABEL = {
    "stock":         "stock, memory pool, no sanitizer",
    "memcheck":      "memory pool under memcheck",
    "exact":         "exact-size cudaMalloc under memcheck",
    "guard":         "guard page on src1 and ids_dst, no sanitizer",
    "guard_ids_dst": "guard page on ids_dst only, no sanitizer",
    "guard_src1":    "guard page on src1 only, no sanitizer",
}
VARIANT_LABEL = {
    "ne11":      "ne11 (#24127)",
    "ne12":      "ne12 (#29941, master)",
    "p29953":    "#29953",
    "p27044":    "ne12*n_expert_used (#27044)",
    "successor": "widest tile (#448)",
    "s448p":     "widest padded tile (#448 amended)",
    "p29953fix": "#29953 + amendment",
}
VARIANT_ORDER = ["ne11", "ne12", "p29953", "p27044", "successor", "s448p", "p29953fix"]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = []
    for path in sys.argv[1:]:
        with open(path) as f:
            rows += [r for r in csv.DictReader(f, delimiter="\t") if r.get("case")]

    cases = list(dict.fromkeys(r["case"] for r in rows))
    modes = [m for m in MODE_LABEL if any(r["mode"] == m for r in rows)]
    seen = list(dict.fromkeys(r["variant"] for r in rows))
    unknown = [v for v in seen if v not in VARIANT_LABEL]
    if unknown:
        sys.exit("variant(s) %s are in the data but not in VARIANT_LABEL: add them or the column is dropped"
                 % ", ".join(unknown))
    variants = [v for v in VARIANT_ORDER if v in seen]
    cell = collections.defaultdict(list)
    for r in rows:
        cell[(r["case"], r["variant"], r["mode"])].append(r)

    def fmt(rs):
        if not rs:
            return "-"
        npass = sum(r["passed"] == "1" for r in rs)
        out = "%d/%d" % (npass, len(rs))
        errs = {r["memcheck_errors"] for r in rs if r["memcheck_errors"]}
        if errs:
            out += " E=" + "/".join(sorted(errs, key=lambda e: int(e)))
        if npass < len(rs) and any(r["illegal_access"] not in ("", "0") for r in rs):
            out += " abort"
        return out

    for m in modes:
        print("### %s\n" % MODE_LABEL[m])
        print("| case | " + " | ".join(VARIANT_LABEL[v] for v in variants) + " |")
        print("|---" * (len(variants) + 1) + "|")
        for c in cases:
            if not any(cell[(c, v, m)] for v in variants):
                continue
            print("| `%s` | " % c + " | ".join(fmt(cell[(c, v, m)]) for v in variants) + " |")
        print()


if __name__ == "__main__":
    main()
