#!/usr/bin/env python3
"""Rewrite a llama.cpp b11081 checkout's mmq.cu into one form of the fork's compat 903, optionally with the
debug allocator. The b11081 counterpart of mmq-variant.py, which targets dd266785c.

    mmq-variant-b11081.py <llama.cpp checkout at b11081> <none|main|widest|amended> [--debug]

Forms, each the 903 patch as committed at a fixed SHA of this repository (read with git, so no copies drift):

  none     b11081 as pinned: the ids branch pads get_J_max(ne11), #24127's rule
  main     903 at de1c32af3 (main when gfx1151 was measured): get_J_max(ne12*n_expert_used), #27044's rule
  widest   903 at 57f9fbc871: pad for the widest tile that has a config, get_J_max(type, fallback, cc, 512)
  amended  903 at bfc3fe487 (unchanged through 1ecdb175c): pad for the widest padded tile

--debug installs mmq-debug-alloc-b11081.cuh as mmq-debug-alloc.cuh and adds the MMQ_DEBUG_ALLOC / MMQ_DEBUG_PRINT
hooks to the ids branch -- mmq-variant.py's debug() as of bfc3fe487, with b11081's anchors (no prec_src1). Each
edit asserts its anchor occurs exactly once, so a moved tree fails loudly instead of yielding a silently different
variant. mmq.cu is reset to b11081 first, so the script can be rerun on the same checkout.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FORMS = {"main": "de1c32af3", "widest": "57f9fbc871", "amended": "bfc3fe487"}
PATCH = "llama/compat/903-fix-mmq-ids-padding.patch"

src, v = sys.argv[1], sys.argv[2]
dbg = "--debug" in sys.argv[3:]
if v != "none" and v not in FORMS:
    sys.exit("unknown form %s" % v)
path = os.path.join(src, "ggml/src/ggml-cuda/mmq.cu")
subprocess.run(["git", "checkout", "b11081", "--", "ggml/src/ggml-cuda/mmq.cu"], cwd=src, check=True)
if v != "none":
    patch = subprocess.run(["git", "show", "%s:%s" % (FORMS[v], PATCH)], cwd=HERE, check=True,
                           capture_output=True).stdout
    subprocess.run(["git", "apply", "-"], cwd=src, input=patch, check=True)
s = open(path).read()


def rep(old, new):
    global s
    n = s.count(old)
    if n != 1:
        sys.exit("anchor found %d times, expected once: %r" % (n, old[:90]))
    s = s.replace(old, new)


if dbg:
    shutil.copy(os.path.join(HERE, "mmq-debug-alloc-b11081.cuh"),
                os.path.join(src, "ggml/src/ggml-cuda/mmq-debug-alloc.cuh"))
    rep('#include "mmq.cuh"\n', '#include "mmq.cuh"\n#include "mmq-debug-alloc.cuh"\n')
    IDS_DST_RE = re.compile(r"^    ggml_cuda_pool_alloc<int32_t> ids_dst\(ctx\.pool\(\), (?P<n>[^)]+)\);$", re.M)
    m = IDS_DST_RE.search(s)
    rep(m.group(0), """    const size_t       mmq_dbg_ids_dst_n = %s;
    const mmq_dbg_mode mmq_dbg_m_ids     = mmq_dbg_get_mode(MMQ_DBG_IDS_DST);
    const mmq_dbg_mode mmq_dbg_m_src1    = mmq_dbg_get_mode(MMQ_DBG_SRC1);
    ggml_cuda_pool_alloc<int32_t> ids_dst_pool(ctx.pool());
    mmq_dbg_buf mmq_dbg_ids_dst;
    if (mmq_dbg_m_ids) {
        mmq_dbg_ids_dst = mmq_dbg_alloc(mmq_dbg_m_ids, mmq_dbg_ids_dst_n*sizeof(int32_t), sizeof(int32_t));
    } else {
        ids_dst_pool.alloc(mmq_dbg_ids_dst_n);
    }
    mmq_dbg_ptr<int32_t> ids_dst = { mmq_dbg_m_ids ? (int32_t *) mmq_dbg_ids_dst.ptr : ids_dst_pool.get() };""" % m.group("n"))
    rep("\n    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);\n"
        "    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());\n", """
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    mmq_dbg_buf mmq_dbg_src1;
    if (mmq_dbg_m_src1) {
        mmq_dbg_src1 = mmq_dbg_alloc(mmq_dbg_m_src1, nbytes_src1_q8_1, 128);
    } else {
        src1_q8_1_pool.alloc(nbytes_src1_q8_1);
    }
    mmq_dbg_ptr<char> src1_q8_1 = { mmq_dbg_m_src1 ? (char *) mmq_dbg_src1.ptr : src1_q8_1_pool.get() };
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    mmq_dbg_report(src0->type, fallback, cc,
        ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo, ne11, ne12, n_expert_used, ne02,
        nbytes_src1_q8_1 - ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block,
        mmq_dbg_ids_dst_n, ne_get_rows);
""")
    rep("    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);\n}\n\nbool ggml_cuda_should_use_mmq(",
        """    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream);
    if (mmq_dbg_m_ids || mmq_dbg_m_src1) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        mmq_dbg_free(mmq_dbg_src1);
        mmq_dbg_free(mmq_dbg_ids_dst);
    }
}

bool ggml_cuda_should_use_mmq(""")
open(path, "w").write(s)
print("%s: 903=%s%s" % (path, v, " + MMQ_DEBUG_ALLOC/PRINT" if dbg else ""))
