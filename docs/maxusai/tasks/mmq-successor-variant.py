#!/usr/bin/env python3
"""Rewrite llama.cpp master's ggml/src/ggml-cuda/mmq.cu into one padding variant, optionally with the debug switch.

    mmq-successor-variant.py MMQ_CU VARIANT [--exact]

VARIANT is one of:
  pre29941   the ids branch pads get_J_max(ne11): the code #29847 faulted on (b9992 .. dd266785c^)
  master     unchanged: get_J_max(ne12), #29941 (dd266785c)
  27044      get_J_max(ne12*n_expert_used): the line llama.cpp#27044 proposed, the fork's compat 903
  successor  both branches pad for the widest tile, get_J_max(type, fallback, cc, 512), and ids_dst and the NVFP4
             src1_scale are padded by the same number of entries: mmq-successor-fix.patch

--exact adds a debug switch, not for upstream: with MMQ_EXACT=1 in the environment, ids_dst and src1_q8_1 get their
own exact-size cudaMalloc instead of the memory pool, so compute-sanitizer flags a read past either one. With the
pool, such a read lands in memory the pool has already mapped, and memcheck cannot see it.

The file is edited in place. Every edit asserts that its anchor occurs exactly once, so a master that has moved
fails loudly instead of yielding a silently different variant.
"""
import sys

IDS_LINE = "        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) * sizeof(block_q8_1_mmq);"
DENSE_LINE = "            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);"
IDS_DST = "    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);"
# The ids branch's allocation, at four spaces after a newline; the dense branch's, at eight, does not match.
SRC1 = "\n    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);\n    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());\n"
SCALE_IDS = "        src1_scale.alloc(ne12*n_expert_used);"
SCALE_DENSE = "            src1_scale.alloc(ne13*ne12*ne11);"
LAUNCH = "    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);\n}\n\nbool ggml_cuda_should_use_mmq("
FALLBACK = "    const bool fallback = ne01 % 128 != 0;\n"


def rep(s, old, new):
    n = s.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times, expected once: {old[:80]!r}")
    return s.replace(old, new)


def variant(s, v):
    if v == "master":
        return s
    if v == "pre29941":
        return rep(s, IDS_LINE, IDS_LINE.replace("cc, ne12)", "cc, ne11)"))
    if v == "27044":
        return rep(s, IDS_LINE, IDS_LINE.replace("cc, ne12)", "cc, ne12*n_expert_used)"))
    if v == "successor":
        s = rep(s, FALLBACK, FALLBACK + """
    // MMQ loads src1 and ids_dst in whole tiles of J columns, and the stream-k fixup path reads the NVFP4 y scales
    // the same way, so the last tile can read up to J - 1 columns past the data: for MUL_MAT_ID the last expert's
    // last tile may hold a single row. mul_mat_q_switch_J rounds the batch up to J, while get_J_max() of the batch
    // rounds down. So pad for the widest tile that has a config, as get_mmq_x_max_host() did before #24127.
    const int64_t J_pad = ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, 512);
""")
        s = rep(s, DENSE_LINE, "            J_pad * sizeof(block_q8_1_mmq);")
        s = rep(s, IDS_LINE, "        J_pad * sizeof(block_q8_1_mmq);")
        # NVFP4 with native FP4: the stream-k fixup write-back reads y_scale for all J columns of the last tile
        s = rep(s, SCALE_IDS, SCALE_IDS.replace("n_expert_used);", "n_expert_used + J_pad);"))
        s = rep(s, SCALE_DENSE, SCALE_DENSE.replace("ne11);", "ne11 + J_pad);"))
        return rep(s, IDS_DST, IDS_DST.replace("ne_get_rows);", "ne_get_rows + J_pad);"))
    sys.exit(f"unknown variant {v}")


def exact(s):
    s = rep(s, '#include "mmq.cuh"\n', '#include "mmq.cuh"\n#include <cstdlib>\n')
    ids_dst_n = "ne_get_rows + J_pad" if "ne_get_rows + J_pad);" in s else "ne_get_rows"
    old_ids_dst = f"    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), {ids_dst_n});"
    s = rep(s, old_ids_dst, f"""    // MMQ_EXACT (debug, not for upstream): MMQ_EXACT=1 gives ids_dst and src1_q8_1 their own exact-size cudaMalloc,
    // so compute-sanitizer flags a read past either instead of landing in memory the pool has already mapped.
    static const bool mmq_exact = getenv("MMQ_EXACT") != nullptr;
    const size_t mmq_exact_ids_dst_n = {ids_dst_n};
    int32_t * mmq_exact_ids_dst = nullptr;
    ggml_cuda_pool_alloc<int32_t> ids_dst_pool(ctx.pool());
    if (mmq_exact) {{ CUDA_CHECK(cudaMalloc(&mmq_exact_ids_dst, mmq_exact_ids_dst_n*sizeof(int32_t))); }} else {{ ids_dst_pool.alloc(mmq_exact_ids_dst_n); }}
    struct {{ int32_t * p; int32_t * get() const {{ return p; }} }} ids_dst = {{ mmq_exact ? mmq_exact_ids_dst : ids_dst_pool.get() }};""")
    s = rep(s, SRC1, """
    char * mmq_exact_src1 = nullptr;
    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx.pool());
    if (mmq_exact) { CUDA_CHECK(cudaMalloc(&mmq_exact_src1, nbytes_src1_q8_1)); } else { src1_q8_1_pool.alloc(nbytes_src1_q8_1); }
    struct { char * p; char * get() const { return p; } } src1_q8_1 = { mmq_exact ? mmq_exact_src1 : src1_q8_1_pool.get() };
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
""")
    return rep(s, LAUNCH, """    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
    if (mmq_exact) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(mmq_exact_src1));
        CUDA_CHECK(cudaFree(mmq_exact_ids_dst));
    }
}

bool ggml_cuda_should_use_mmq(""")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    path, v = sys.argv[1], sys.argv[2]
    s = open(path).read()
    s = variant(s, v)
    if "--exact" in sys.argv[3:]:
        s = exact(s)
    open(path, "w").write(s)
    print(f"{path}: variant {v}{' + MMQ_EXACT switch' if '--exact' in sys.argv[3:] else ''}")


if __name__ == "__main__":
    main()
