// A compile-time guard for the MMQ src1 tail padding: one static_assert per (config table, ggml type), each
// requiring that every config's y load fits the padding a launch with it reserves. GPU-free and test-free: it fails
// the BUILD, on every architecture's table at once, in any CUDA or HIP compile that includes it.
//
// The requirement is written from the kernel, not from the padding. mul_mat_q_process_tile loads a tile's y data as
//     for (l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += nwarps*warp_size) tile_y[l0 + tid] = by0[l0 + tid];
// with no bound on l, so it copies GGML_PAD(J*MMQ_TILE_Y_K, nthreads) ints from the tile's first column. A last
// tile holding one column reads that less one block past the data, and every launchable config can end there.
//
// This file is a prototype: it carries its own copy of each padding rule, chosen with -DMMQ_GUARD_RULE=
//   0  padded tile of the launched config   llama.cpp#29953 as of 3070d927f            (default)
//   1  J blocks                             #29953 as first posted -- a re-tightening
//   2  widest tile of the (type, fallback)  compat 903 before its amendment
//   3  widest padded tile, floored          compat 903 as amended
// To guard shipped code, the rule must instead be the helper the allocation itself calls; see compat 903's guard.
//
// One assertion per (table, type) on purpose: one per table puts up to 352 configs, each a chain of CASE tests,
// in one constant evaluation, which can exceed clang's -fconstexpr-steps (1,048,576) -- and that fails the
// assertion with "not an integral constant expression", indistinguishable at a glance from a real shortfall.
//
//   hipcc -std=c++17 -DGGML_USE_HIP --offload-arch=gfx1151 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//         [-DMMQ_GUARD_RULE=N] -c mmq-padding-guard.cu -o /dev/null
// nvcc is untested: its front end has its own constant-evaluation limits.
#include "ggml/src/ggml-cuda/mmq.cuh"
#include <utility>

#ifndef MMQ_GUARD_RULE
#define MMQ_GUARD_RULE 0
#endif

// bytes the y load copies from the first column of a tile, from the load loop's own constants
static constexpr size_t mmq_guard_y_load(const ggml_cuda_mmq_config & c) {
    return GGML_PAD(c.J*MMQ_TILE_Y_K, c.nthreads)*sizeof(int);
}

#define MMQ_PADDING_GUARD(TABLE)                                                                                       \
    static constexpr bool mmq_padding_ok_##TABLE(const int t) {                                                         \
        constexpr size_t B = sizeof(block_q8_1_mmq);                                                                    \
        for (int fb = 0; fb < 2; ++fb) {                                                                                \
            size_t wide = 0, wide_padded = 0;                                                                           \
            if (MMQ_GUARD_RULE >= 2) for (int J = 8; J <= 128; J += 8) {                                                \
                const ggml_cuda_mmq_config w = ggml_cuda_mmq_get_config_##TABLE((ggml_type) t, J, fb);                  \
                if (w.type == GGML_TYPE_COUNT) continue;                                                                \
                if (w.J*B > wide) wide = w.J*B;                                                                         \
                if (mmq_guard_y_load(w)/B*B > wide_padded) wide_padded = mmq_guard_y_load(w)/B*B;                       \
            }                                                                                                           \
            for (int J = 8; J <= 128; J += 8) {                                                                         \
                const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config_##TABLE((ggml_type) t, J, fb);                  \
                if (c.type == GGML_TYPE_COUNT) continue;                                                                \
                const size_t pad = MMQ_GUARD_RULE == 0 ? GGML_PAD(c.J*B, c.nthreads*sizeof(int))                        \
                                 : MMQ_GUARD_RULE == 1 ? c.J*B                                                          \
                                 : MMQ_GUARD_RULE == 2 ? wide : wide_padded;                                            \
                if (pad + B < mmq_guard_y_load(c)) return false;                                                        \
            }                                                                                                           \
        }                                                                                                               \
        return true;                                                                                                    \
    }                                                                                                                   \
    template <int T> struct mmq_padding_guard_##TABLE {                                                                 \
        static_assert(mmq_padding_ok_##TABLE(T), "MMQ src1 padding is short for a config in the " #TABLE " table");    \
    };                                                                                                                  \
    template <int... Ts> static constexpr size_t mmq_padding_all_##TABLE(std::integer_sequence<int, Ts...>) {           \
        return (sizeof(mmq_padding_guard_##TABLE<Ts>) + ... + 0);                                                       \
    }                                                                                                                   \
    static_assert(mmq_padding_all_##TABLE(std::make_integer_sequence<int, GGML_TYPE_COUNT>()) > 0, "");

MMQ_PADDING_GUARD(pascal_older)
MMQ_PADDING_GUARD(pascal_dp4a)
MMQ_PADDING_GUARD(ampere)
MMQ_PADDING_GUARD(blackwell)
MMQ_PADDING_GUARD(gcn)
MMQ_PADDING_GUARD(cdna)
MMQ_PADDING_GUARD(rdna2)
MMQ_PADDING_GUARD(rdna3)
MMQ_PADDING_GUARD(rdna3_5)
MMQ_PADDING_GUARD(rdna4)
