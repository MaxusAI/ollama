#!/usr/bin/env python3
"""mmq-variant.py for the v0.40.16 fold's tree: b12177 plus the fold's ggml-cuda patches 805 and 903. As
mmq-variant-0413.py, with PREC re-anchored on b12177's line, where the fused gate carries its own src1 precision.
Its notes at the v0.40.13 fold (b11963 + 805, 903, 908) follow.

    mmq-variant-0416.py <tree> <pr462|pr462jb|pr462ids0> [--debug]

Loads mmq-variant.py and changes only what b11963 moved; its variants, its MMQ_DEBUG_ALLOC rewrite and its
exactly-once anchors are reused as they are:

  - c49edf096 (fused gate/up + GLU) makes prec_src1 depend on the fused gate, so the line the debug rewrite anchors
    on reads `gate ? GGML_PREC_Q8 : ggml_cuda_mmq_get_prec_src1(...)`. Both branches still take src1, src1_scale
    and ids_dst from the same three allocations, fused or not, so the rewrite covers the fused path unchanged.
  - The debug build also prints `MMQ_FUSED <0|1>` ahead of each MMQ_DEBUG_PRINT line, so a log shows whether a
    case reached the fused kernel.

Then it checks the result, not just the anchors: each variant must carry its own padding expressions and none of
the others', so a control whose mutation did not land fails here instead of passing vacuously.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mmq_variant", os.path.join(HERE, "mmq-variant.py"))
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)

mv.PREC = ("    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0->type, gate ? fusion->prec_src1 : "
           "ggml_prec_get_src(dst, 1), cc);\n")

PAD_903 = "const size_t src1_q8_1_padding = nbytes_y_tile_best;"
PAD_JB  = "const size_t src1_q8_1_padding = J_best*sizeof(block_q8_1_mmq);"
IDS_PAD = "ne_get_rows + J_best-1"
IDS0 = {False: "ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);",
        True:  "const size_t mmq_dbg_ids_dst_n = ne_get_rows;"}

_debug = mv.debug


def debug(s):
    s = _debug(s)
    for prefix in ('        mmq_dbg_report("DENSE",', '    mmq_dbg_report("IDS",'):
        indent = prefix[:len(prefix) - len(prefix.lstrip())]
        s = mv.rep(s, prefix, '%sif (mmq_dbg_print()) {\n%s    fprintf(stderr, "MMQ_FUSED %%d ", (int) (gate != nullptr));\n'
                              '%s}\n%s' % (indent, indent, indent, prefix))
    return s


def check(s, v, dbg):
    # string -> how many times it must occur
    if v == "pr462":
        want, absent = {PAD_903: 1, IDS_PAD: 1}, [PAD_JB]
    elif v == "pr462jb":
        want, absent = {PAD_JB: 1, IDS_PAD: 1}, [PAD_903]
    elif v == "pr462ids0":
        want, absent = {PAD_903: 1, IDS0[dbg]: 1}, [IDS_PAD]
    else:
        sys.exit("mmq-variant-0416.py takes pr462, pr462jb or pr462ids0, not %s" % v)
    if dbg:
        # one include; the dense and the ids branch each route src1 and print; only the ids branch has ids_dst
        want.update({'#include "mmq-debug-alloc.cuh"': 1, "MMQ_FUSED": 2,
                     "mmq_dbg_alloc(mmq_dbg_m_src1": 2, "mmq_dbg_alloc(mmq_dbg_m_ids": 1})
    for w, n in want.items():
        if s.count(w) != n:
            sys.exit("variant %s%s: expected %r %d times in mmq.cu, found it %d times"
                     % (v, " --debug" if dbg else "", w, n, s.count(w)))
    for a in absent:
        if a in s:
            sys.exit("variant %s%s: %r is still in mmq.cu, so the mutation did not apply" % (v, " --debug" if dbg else "", a))


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, v = sys.argv[1], sys.argv[2]
    dbg = "--debug" in sys.argv[3:]
    path = os.path.join(src, "ggml/src/ggml-cuda/mmq.cu")
    base = open(path).read()
    if base.count(PAD_903) != 1 or base.count(IDS_PAD) != 1 or "nbytes_y_tile_best" not in base:
        sys.exit("%s is not b12177 + the v0.40.16 fold's 903" % path)
    mv.shutil.copy(os.path.join(HERE, "mmq-debug-alloc.cuh"), os.path.join(src, "ggml/src/ggml-cuda/"))
    s = mv.variant(src, path, v)
    if dbg:
        s = debug(s)
    check(s, v, dbg)
    if not dbg and v == "pr462" and s != base:
        sys.exit("pr462 without --debug must be the tree's mmq.cu byte for byte")
    open(path, "w").write(s)
    print("%s: variant %s%s (checked)" % (path, v, " + MMQ_DEBUG_ALLOC" if dbg else ""))


if __name__ == "__main__":
    main()
