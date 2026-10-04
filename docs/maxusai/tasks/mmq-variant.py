#!/usr/bin/env python3
"""Rewrite llama.cpp's ggml/src/ggml-cuda/mmq.cu into one MMQ padding variant, optionally with the debug allocator.

    mmq-variant.py <llama.cpp checkout> <variant> [--debug]

The checkout must be at dd266785c (#29941 merged) or a commit with the same mmq.cu, clean. Variants:

  ne11        the ids branch pads get_J_max(ne11): the code #29847 faulted on (b9992 .. dd266785c^)
  ne12        unchanged master: get_J_max(ne12), i.e. llama.cpp#29941 as merged (dd266785c)
  p29953      llama.cpp#29953 applied as published: the tile width is chosen once, before the allocation, and
              both branches pad by exactly that J_best. Touches mmq.cu and mmq.cuh.
  p29953fix   #29953 plus the two amendments it needs: pad src1 by the padded y tile the load actually copies,
              GGML_PAD(J_best*sizeof(block_q8_1_mmq), nthreads*sizeof(int)), rather than J_best blocks; and pad
              ids_dst (and the NVFP4 y scales) by J_best, which no rule so far does.
  p27044      get_J_max(ne12*n_expert_used): the line llama.cpp#27044 proposed, our compat 903 until now
  successor   #448 as published: both branches pad for the widest tile that has a config,
              get_J_max(type, fallback, cc, 512), and ids_dst and the NVFP4 src1_scale by the same count
  s448p       #448 amended: pad for the widest *padded* tile instead. A tile's y load copies the whole padded
              shared-memory y tile from global memory, GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int))
              bytes, which is more than J blocks when nthreads*4 does not divide J*144

--debug adds the MMQ_DEBUG_ALLOC switch (mmq-debug-alloc.cuh, debug only, never shipped): the ids branch's
src1_q8_1 and ids_dst come from their own allocation instead of the memory pool, so a read past either is
observable -- `exact` makes compute-sanitizer report it, `guard` makes it fault. Without it the read lands in
memory the pool has already mapped and nothing sees it.

Each edit asserts its anchor occurs exactly once, so a moved master fails loudly instead of yielding a silently
different variant.
"""
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

IDS_LINE   = "        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) * sizeof(block_q8_1_mmq);"
DENSE_LINE = "            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);"
FALLBACK   = "    const bool fallback = ne01 % 128 != 0;\n"
PREC       = "    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);\n"
SCALE_IDS   = "        src1_scale.alloc(ne12*n_expert_used);"
SCALE_DENSE = "            src1_scale.alloc(ne13*ne12*ne11);"

# The ids branch's allocations. The dense branch's are indented four spaces deeper and do not match.
SCALE_RE = re.compile(r"src1_scale\.alloc\(([^)]+)\);")
IDS_DST_RE = re.compile(r"^    ggml_cuda_pool_alloc<int32_t> ids_dst\(ctx\.pool\(\), (?P<n>[^)]+)\);.*$", re.M)
SRC1 = ("\n    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);\n"
        "    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());\n")
SRC1_DENSE = ("\n        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);\n"
              "        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());\n")
LAUNCH_DENSE = ("        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);\n"
                "        return;\n    }\n")
LAUNCH_DENSE_FREE = LAUNCH_DENSE
LAUNCH = "    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);\n}\n\nbool ggml_cuda_should_use_mmq("

S448P_PAD = """
    // MMQ loads src1 and ids_dst in whole tiles of J columns, and the stream-k fixup path reads the NVFP4 y
    // scales the same way. A tile's y load copies the whole padded shared-memory y tile from global memory --
    // GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int)) bytes from the tile's first column, the same
    // expression mmq_get_nbytes_shared() uses to size the tile it copies into -- so when the last tile holds a
    // single column all but one block of that is past the data: for MUL_MAT_ID the last expert's last tile may
    // hold a single row, at any batch size. mul_mat_q_switch_J rounds the batch up to J while get_J_max() of the
    // batch rounds down, so pad for the widest tile any launch could pick.
    size_t nbytes_pad_y = 0;
    for (int J_cand = 8; J_cand <= 128; J_cand += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(src0->type, J_cand, fallback, cc, prec_src1);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }
        const size_t nbytes_y = GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));
        if (nbytes_y > nbytes_pad_y) {
            nbytes_pad_y = nbytes_y;
        }
    }
    const int64_t J_pad = nbytes_pad_y / sizeof(block_q8_1_mmq);
"""

SUCCESSOR_PAD = """
    // MMQ loads src1 and ids_dst in whole tiles of J columns, and the stream-k fixup path reads the NVFP4 y scales
    // the same way, so the last tile can read up to J - 1 columns past the data: for MUL_MAT_ID the last expert's
    // last tile may hold a single row. mul_mat_q_switch_J rounds the batch up to J, while get_J_max() of the batch
    // rounds down. So pad for the widest tile that has a config, as get_mmq_x_max_host() did before #24127.
    const int64_t J_pad = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, 512);
"""


def rep(s, old, new):
    n = s.count(old)
    if n != 1:
        sys.exit("anchor found %d times, expected once: %r" % (n, old[:90]))
    return s.replace(old, new)


def variant(src, path, v):
    s = open(path).read()
    if v == "ne12":
        return s
    if v == "head":
        return s      # the PR's own head as fetched: already pads the tile and ids_dst, but not the y scales
    if v == "headyscale":
        m = SCALE_RE.findall(s)
        if len(m) != 2:
            sys.exit("expected two src1_scale.alloc calls, found %d" % len(m))
        for arg in m:
            s = rep(s, "src1_scale.alloc(%s);" % arg,
                       "src1_scale.alloc(%s + J_best-1);" % arg)
        return s
    if v == "ne11":
        return rep(s, IDS_LINE, IDS_LINE.replace("cc, ne12)", "cc, ne11)"))
    if v == "p27044":
        return rep(s, IDS_LINE, IDS_LINE.replace("cc, ne12)", "cc, ne12*n_expert_used)"))
    if v in ("p29953", "p29953fix"):
        patch = os.path.join(HERE, "p29953.patch")
        if not os.path.exists(patch):
            sys.exit("%s is missing; fetch it with\n"
                     "  gh pr diff 29953 --repo ggml-org/llama.cpp > %s" % (patch, patch))
        subprocess.run(["git", "apply", patch], cwd=src, check=True)
        s = open(path).read()
        if v == "p29953":
            return s
        # Carry the chosen config's thread count out of the loop: the y tile load copies
        # GGML_PAD(J*sizeof(block_q8_1_mmq), nthreads*sizeof(int)) bytes from the tile's first column, which is
        # what mmq_get_nbytes_shared() sizes the shared-memory tile it lands in, and more than J blocks whenever
        # nthreads*sizeof(int) does not divide J*sizeof(block_q8_1_mmq).
        s = rep(s, "    int J_best = 0;\n", "    int    J_best       = 0;\n    size_t nbytes_pad_y = 0;\n")
        s = rep(s, """            if (ntiles_x < ntiles_J_best) {
                J_best = J;
                ntiles_J_best = ntiles_x;
            }""", """            if (ntiles_x < ntiles_J_best) {
                J_best        = J;
                ntiles_J_best = ntiles_x;
                nbytes_pad_y  = GGML_PAD(config.J*sizeof(block_q8_1_mmq), config.nthreads*sizeof(int));
            }""")
        s = rep(s, "            J_best * sizeof(block_q8_1_mmq);", "            nbytes_pad_y;")   # dense branch
        s = rep(s, "        J_best * sizeof(block_q8_1_mmq);", "        nbytes_pad_y;")           # ids branch
        if "J_best * sizeof(block_q8_1_mmq)" in s:
            sys.exit("a J_best padding term was left behind")
        # ids_dst: a tile loads J entries from the expert's first row with no bound, so the last expert's last
        # tile reads up to J - 1 entries past ne_get_rows. The NVFP4 y scales are read the same way by the
        # stream-k fixup write-back.
        m = IDS_DST_RE.search(s)
        s = rep(s, m.group(0), m.group(0).replace("ne_get_rows);", "ne_get_rows + J_best);"))
        s = rep(s, SCALE_IDS, SCALE_IDS.replace("n_expert_used);", "n_expert_used + J_best);"))
        return rep(s, SCALE_DENSE, SCALE_DENSE.replace("ne11);", "ne11 + J_best);"))
    if v in ("successor", "s448p"):
        if v == "successor":
            s = rep(s, FALLBACK, FALLBACK + SUCCESSOR_PAD)
        else:
            s = rep(s, PREC, PREC + S448P_PAD)
        s = rep(s, DENSE_LINE, "            J_pad * sizeof(block_q8_1_mmq);")
        s = rep(s, IDS_LINE, "        J_pad * sizeof(block_q8_1_mmq);")
        # NVFP4 with native FP4: the stream-k fixup write-back reads y_scale for all J columns of the last tile.
        s = rep(s, SCALE_IDS, SCALE_IDS.replace("n_expert_used);", "n_expert_used + J_pad);"))
        s = rep(s, SCALE_DENSE, SCALE_DENSE.replace("ne11);", "ne11 + J_pad);"))
        m = IDS_DST_RE.search(s)
        return rep(s, m.group(0), m.group(0).replace("%s);" % m.group("n"), "%s + J_pad);" % m.group("n")))
    sys.exit("unknown variant %s" % v)


def debug(s):
    s = rep(s, '#include "mmq.cuh"\n', '#include "mmq.cuh"\n#include "mmq-debug-alloc.cuh"\n')

    # Both branches need these, and the dense branch comes first. Anchored on prec_src1 rather than fallback
    # because #29953 rewrites the fallback line into ggml_cuda_mmq_needs_fallback(ne01).
    s = rep(s, PREC, PREC + """
    const mmq_dbg_mode mmq_dbg_m_ids  = mmq_dbg_get_mode(MMQ_DBG_IDS_DST);
    const mmq_dbg_mode mmq_dbg_m_src1 = mmq_dbg_get_mode(MMQ_DBG_SRC1);
    const mmq_dbg_mode mmq_dbg_m_scale = mmq_dbg_get_mode(MMQ_DBG_SRC1_SCALE);
""")

    # The dense branch: src1 only, and it returns before the ids branch's teardown.
    s = rep(s, SRC1_DENSE, """
        ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
        mmq_dbg_buf mmq_dbg_src1_dense;
        if (mmq_dbg_m_src1) {
            mmq_dbg_src1_dense = mmq_dbg_alloc(mmq_dbg_m_src1, nbytes_src1_q8_1, 128);
        } else {
            src1_q8_1_pool.alloc(nbytes_src1_q8_1);
        }
        mmq_dbg_ptr<char> src1_q8_1 = { mmq_dbg_m_src1 ? (char *) mmq_dbg_src1_dense.ptr : src1_q8_1_pool.get() };
        ggml_cuda_pool_alloc<float> src1_scale_pool(ctx.pool());
        mmq_dbg_buf mmq_dbg_scale_dense;
        mmq_dbg_ptr<float> src1_scale = { nullptr };
        mmq_dbg_report("DENSE", src0->type, fallback, prec_src1, cc,
            ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo, ne11, ne12, 0, ne02,
            nbytes_src1_q8_1 - ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block, -1, 0);
""")
    # teardown is added after the scale allocation is routed, below

    m = IDS_DST_RE.search(s)
    if m is None:
        sys.exit("could not find the ids branch's ids_dst allocation")
    # ids_dst is read with scalar int loads, so four bytes of alignment is all the kernel needs: the guard is tight
    # to the entry and a single int past the end faults.
    s = rep(s, m.group(0), """    const size_t mmq_dbg_ids_dst_n = %s;
    ggml_cuda_pool_alloc<int32_t> ids_dst_pool(ctx.pool());
    mmq_dbg_buf mmq_dbg_ids_dst;
    if (mmq_dbg_m_ids) {
        mmq_dbg_ids_dst = mmq_dbg_alloc(mmq_dbg_m_ids, mmq_dbg_ids_dst_n*sizeof(int32_t), sizeof(int32_t));
    } else {
        ids_dst_pool.alloc(mmq_dbg_ids_dst_n);
    }
    mmq_dbg_ptr<int32_t> ids_dst = { mmq_dbg_m_ids ? (int32_t *) mmq_dbg_ids_dst.ptr : ids_dst_pool.get() };""" % m.group("n"))

    # src1_q8_1 is read as whole block_q8_1_mmq tiles through vector loads; 128 bytes keeps those legal and still
    # leaves less slack than the 144-byte block an over-read would take.
    s = rep(s, SRC1, """
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    mmq_dbg_buf mmq_dbg_src1;
    if (mmq_dbg_m_src1) {
        mmq_dbg_src1 = mmq_dbg_alloc(mmq_dbg_m_src1, nbytes_src1_q8_1, 128);
    } else {
        src1_q8_1_pool.alloc(nbytes_src1_q8_1);
    }
    mmq_dbg_ptr<char> src1_q8_1 = { mmq_dbg_m_src1 ? (char *) mmq_dbg_src1.ptr : src1_q8_1_pool.get() };
    ggml_cuda_pool_alloc<float> src1_scale_pool(ctx.pool());
    mmq_dbg_buf mmq_dbg_scale;
    mmq_dbg_ptr<float> src1_scale = { nullptr };
    mmq_dbg_report("IDS", src0->type, fallback, prec_src1, cc,
        ggml_cuda_info().devices[ggml_cuda_get_device()].smpbo, ne11, ne12, n_expert_used, ne02,
        nbytes_src1_q8_1 - ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block,
        mmq_dbg_ids_dst_n, ne_get_rows);
""")

    # Route both src1_scale.alloc() calls through the debug allocator. Scalar float loads, so four bytes of
    # alignment is all the kernel needs and a single float past the end faults.
    for indent, var in ((" " * 12, "mmq_dbg_scale_dense"), (" " * 8, "mmq_dbg_scale")):
        m = re.search(r"^%ssrc1_scale\.alloc\(([^)]+)\);.*$" % indent, s, re.M)
        if m is None:
            sys.exit("could not find a src1_scale.alloc at indent %d" % len(indent))
        s = rep(s, m.group(0), """%sconst size_t mmq_dbg_scale_n = %s;
%sif (mmq_dbg_m_scale) {
%s    %s = mmq_dbg_alloc(mmq_dbg_m_scale, mmq_dbg_scale_n*sizeof(float), sizeof(float));
%s    src1_scale.ptr = (float *) %s.ptr;
%s} else {
%s    src1_scale_pool.alloc(mmq_dbg_scale_n);
%s    src1_scale.ptr = src1_scale_pool.get();
%s}
%sif (mmq_dbg_print()) {
%s    fprintf(stderr, "MMQ_SCALE alloc n=%%zu guarded=%%d\\n", (size_t) mmq_dbg_scale_n, (int) (mmq_dbg_m_scale != 0));
%s}""" % (indent, m.group(1), indent, indent, var, indent, var, indent, indent, indent, indent,
                   indent, indent, indent))

    s = rep(s, LAUNCH_DENSE_FREE, """        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        if (mmq_dbg_m_src1 || mmq_dbg_m_scale) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            mmq_dbg_free(mmq_dbg_src1_dense);
            mmq_dbg_free(mmq_dbg_scale_dense);
        }
        return;
    }
""")
    return rep(s, LAUNCH, """    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
    if (mmq_dbg_m_ids || mmq_dbg_m_src1 || mmq_dbg_m_scale) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        mmq_dbg_free(mmq_dbg_src1);
        mmq_dbg_free(mmq_dbg_scale);
        mmq_dbg_free(mmq_dbg_ids_dst);
    }
}

bool ggml_cuda_should_use_mmq(""")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    src, v = sys.argv[1], sys.argv[2]
    dbg = "--debug" in sys.argv[3:]
    path = os.path.join(src, "ggml/src/ggml-cuda/mmq.cu")
    shutil.copy(os.path.join(HERE, "mmq-debug-alloc.cuh"), os.path.join(src, "ggml/src/ggml-cuda/"))
    s = variant(src, path, v)
    if dbg:
        s = debug(s)
    open(path, "w").write(s)
    print("%s: variant %s%s" % (path, v, " + MMQ_DEBUG_ALLOC" if dbg else ""))


if __name__ == "__main__":
    main()
