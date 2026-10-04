// gfx1151 companion to mmq-rules-check.cu (#448, answering #449). Same sweep, same six rules, same tile
// selection, plus the two things that decide whether a shape the check counts is one this device can hit:
//
//   dispatch  ggml_cuda_should_use_mmq() and the MUL_MAT_ID MMVQ batch limit, compiled from the checkout's own
//             mmq.cu / mmvq.cu (cut verbatim by mmq-extract-dispatch.sh into dispatch.inc, not re-typed), with the
//             device's real cc (gfx1151 = OFFSET_AMD + 0x1151) and smpbo (65536, read from hipDeviceProp_t).
//             On RDNA3 q2_K goes to MMQ only with >= 64 experts or <= 128 tokens; otherwise hipBLAS runs it.
//   routing   whether some routing leaves the last non-empty expert's last tile holding one column, which is the
//             case the check assumes. With n_used < n_experts one token can be the only row of the last expert.
//             With n_used == n_experts every expert holds exactly ne12 rows, so the last tile holds
//             (ne12 - 1) % J + 1 columns, and the over-read is that many blocks smaller.
//
// Build from a llama.cpp checkout:
//   mmq-extract-dispatch.sh . > dispatch.inc
//   hipcc -std=c++17 -DGGML_USE_HIP --offload-arch=gfx1151 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda \
//         -I<dir of dispatch.inc> mmq-gfx1151-reach.cu -o mmq-gfx1151-reach
//   ./mmq-gfx1151-reach [--production]   --production also lists K-quant MoE shapes short under main's 903
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }

static ggml_cuda_device_info g_info;
const ggml_cuda_device_info & ggml_cuda_info() { return g_info; }
int ggml_cuda_get_device() { return 0; }

#include "dispatch.inc"

static const size_t BLOCK = sizeof(block_q8_1_mmq);
static const int    CC    = GGML_CUDA_CC_OFFSET_AMD + 0x1151;   // what ggml_cuda_parse_id("gfx1151") returns
static const size_t SMPBO = 65536;                              // hipDeviceProp_t::sharedMemPerBlock on this device

// Bytes one tile's y load copies from its first column.
static size_t tile_bytes(int J, int nthreads) { return GGML_PAD(J*BLOCK, nthreads*sizeof(int)); }

static void launched(ggml_type t, bool fb, int64_t ncols_opt, int * J_out, int * nth_out) {
    int nt_best = INT_MAX;
    *J_out = 0; *nth_out = 0;
    for (int J = 8; J <= 128 && nt_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, CC);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, CC) > SMPBO) continue;
        const int nt = (ncols_opt + c.J - 1) / c.J;
        if (nt < nt_best) { nt_best = nt; *J_out = c.J; *nth_out = c.nthreads; }
    }
}

static void widest(ggml_type t, bool fb, int * J_out, int * pad_out) {
    *J_out = ggml_cuda_mmq_get_J_max(t, fb, CC, 512);
    int pad = 0;
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, CC);
        if (c.type == GGML_TYPE_COUNT) continue;
        const int blocks = (int) ((tile_bytes(c.J, c.nthreads) + BLOCK - 1) / BLOCK);
        if (blocks > pad) pad = blocks;
    }
    *pad_out = pad;
}

enum { R24127, R29941, R29953, R27044, R448, R448P, NR };
static const char * const RN[NR] = {
    "#24127 (ne11), stock b11081", "#29941 (ne12), master", "#29953 (the launched tile)",
    "#27044 (ne12*n_used), main's 903", "#448 (widest tile)", "#448 amended (widest padded tile)"};

static const ggml_type TYPES[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                                  GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
static const char * const TN[] = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"};
static const int NE02S[] = {8, 32, 128, 256};

struct tally {
    long all = 0, reach = 0, real = 0;           // as the PR counts / reaches MMQ / and a routing realises it
    long by_type_all[10] = {}, by_type_real[10] = {};
    long worst_real = 0; char ex[200] = "";      // bytes past the allocation
    void add(int ti, bool short_c1, bool reaches, long over_real, const char * desc) {
        if (short_c1) { ++all; ++by_type_all[ti]; if (reaches) ++reach; }
        if (reaches && over_real > 0) {
            ++real; ++by_type_real[ti];
            if (over_real > worst_real) { worst_real = over_real; snprintf(ex, sizeof(ex), "%s", desc); }
        }
    }
    void print(const char * label) const {
        printf("  %-36s %8ld %8ld %8ld", label, all, reach, real);
        if (real) printf("   worst %ld B past: %s", worst_real, ex);
        printf("\n");
        if (all) {
            printf("  %-36s   by type (all/real):", "");
            for (int i = 0; i < 10; ++i) if (by_type_all[i] || by_type_real[i]) printf(" %s %ld/%ld", TN[i], by_type_all[i], by_type_real[i]);
            printf("\n");
        }
    }
};

int main(int argc, char ** argv) {
    g_info.device_count = 1;
    g_info.devices[0].cc = CC;
    g_info.devices[0].smpbo = SMPBO;
    g_info.devices[0].smpb = SMPBO;
    g_info.devices[0].warp_size = 32;

    const bool prod = argc > 1 && !strcmp(argv[1], "--production");

    tally src1[NR], dst[NR], dense[NR]; long one_all = 0, one_reach = 0;
    long n_ids = 0, n_dense = 0;

    for (int ti = 0; ti < 10; ++ti) for (int fb = 0; fb < 2; ++fb) {
        const ggml_type t = TYPES[ti];
        int J_wide, pad_wide_amended;
        widest(t, fb, &J_wide, &pad_wide_amended);
        const int mmvq_max = get_mmvq_mmid_max_batch(t, CC);

        for (int64_t ne12 = mmvq_max + 1; ne12 <= 1024; ++ne12)
        for (int n_used = 1; n_used <= 16; ++n_used)
        for (int ei = 0; ei < 4; ++ei) {
            const int64_t ne02 = NE02S[ei];
            if (n_used > ne02) continue;
            const int64_t ncols_opt = (ne12*n_used + ne02 - 1) / ne02;
            int J, nth;
            launched(t, fb, ncols_opt, &J, &nth);
            if (J == 0) continue;
            const bool reaches = ggml_cuda_should_use_mmq(t, CC, ne12, ne02);
            // columns in the last non-empty expert's last tile, minimised over routings
            const int64_t c = n_used < ne02 ? 1 : (ne12 - 1) % J + 1;
            const long T = (long) tile_bytes(J, nth);
            for (int bcast = 0; bcast < 2; ++bcast) {
                if (bcast && n_used == 1) continue;
                ++n_ids;
                const int pad[NR] = {
                    ggml_cuda_mmq_get_J_max(t, fb, CC, bcast ? 1 : n_used),
                    ggml_cuda_mmq_get_J_max(t, fb, CC, ne12),
                    J,
                    ggml_cuda_mmq_get_J_max(t, fb, CC, ne12*n_used),
                    J_wide,
                    pad_wide_amended,
                };
                char desc[200];
                for (int r = 0; r < NR; ++r) {
                    snprintf(desc, sizeof(desc), "%s fb=%d ne12=%ld n_used=%d n_exp=%ld b=%d: J=%d nthreads=%d pad=%d blocks",
                             TN[ti], fb, (long) ne12, n_used, (long) ne02, bcast, J, nth, pad[r]);
                    // as the PR counts: one column in the last tile, in blocks
                    const int  need_c1   = (int) ((T + BLOCK - 1) / BLOCK) - 1;
                    const long over_real = T - (long) (c*BLOCK) - (long) pad[r]*BLOCK;
                    src1[r].add(ti, need_c1 > pad[r], reaches, over_real, desc);
                    if (r == R448 && n_used == 1 && need_c1 > pad[r]) { ++one_all; if (reaches) ++one_reach; }
                    const int dst_pad = (r == R448 || r == R448P) ? pad[r] : 0;
                    dst[r].add(ti, J - 1 > dst_pad, reaches, (long) (J - c - dst_pad) * 4, desc);
                }
            }
        }

        for (int64_t ne11 = MMVQ_MAX_BATCH_SIZE + 1; ne11 <= 1024; ++ne11) {
            int J, nth;
            launched(t, fb, ne11, &J, &nth);
            if (J == 0) continue;
            ++n_dense;
            const bool reaches = ggml_cuda_should_use_mmq(t, CC, ne11, 0);
            const int64_t last = ne11 - ((ne11 + J - 1) / J - 1) * J;
            const long T = (long) tile_bytes(J, nth);
            const int pad[NR] = {
                ggml_cuda_mmq_get_J_max(t, fb, CC, ne11), ggml_cuda_mmq_get_J_max(t, fb, CC, ne11), J,
                ggml_cuda_mmq_get_J_max(t, fb, CC, ne11), J_wide, pad_wide_amended,
            };
            char desc[200];
            for (int r = 0; r < NR; ++r) {
                snprintf(desc, sizeof(desc), "%s fb=%d ne11=%ld: J=%d nthreads=%d pad=%d blocks", TN[ti], fb, (long) ne11, J, nth, pad[r]);
                const int need = (int) ((T + BLOCK - 1) / BLOCK) - 1 - (int) (last - 1);
                const long over = T - (long) (last*BLOCK) - (long) pad[r]*BLOCK;
                dense[r].add(ti, need > pad[r], reaches, over, desc);
            }
        }
    }

    printf("gfx1151 (cc 0x%x, smpbo %zu), %ld ids-branch and %ld dense shapes -- the PR's sweep for this arch\n",
           CC & 0xffff, SMPBO, n_ids, n_dense);
    printf("columns: all = short as the PR counts (one column in the last tile); reach = and the build sends the shape\n"
           "         to MMQ; real = reaches MMQ and some routing makes the read leave the allocation\n");
    printf(" src1_q8_1, ids branch:                    %8s %8s %8s\n", "all", "reach", "real");
    for (int r = 0; r < NR; ++r) src1[r].print(RN[r]);
    printf(" ids_dst, ids branch:\n");
    for (int r = 0; r < NR; ++r) dst[r].print(RN[r]);
    printf(" src1, dense branch:\n");
    for (int r = 0; r < NR; ++r) if (r == R24127 || r == R29953 || r == R448 || r == R448P) dense[r].print(RN[r]);

    printf("\n#448 (widest tile), one expert per token: short in %ld shapes, %ld of them reach MMQ\n", one_all, one_reach);
    if (prod) {
        // The shapes production's MoE GGUFs can produce here, under main's 903 (#27044's rule).
        printf("\nproduction-like MUL_MAT_ID shapes short under #27044 (main's 903), n_used 6/8, 128/256 experts, ne12 <= 4096:\n");
        const ggml_type PT[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
        const char * PN[] = {"q4_K", "q5_K", "q6_K", "q8_0", "q4_0"};
        for (int pi = 0; pi < 5; ++pi) for (int fb = 0; fb < 2; ++fb) for (int n_used : {6, 8}) for (int64_t ne02 : {128, 256}) {
            const ggml_type t = PT[pi];
            for (int64_t ne12 = get_mmvq_mmid_max_batch(t, CC) + 1; ne12 <= 4096; ++ne12) {
                if (!ggml_cuda_should_use_mmq(t, CC, ne12, ne02)) continue;
                const int64_t ncols_opt = (ne12*n_used + ne02 - 1) / ne02;
                int J, nth;
                launched(t, fb, ncols_opt, &J, &nth);
                const long T = (long) tile_bytes(J, nth);
                const int pad = ggml_cuda_mmq_get_J_max(t, fb, CC, ne12*n_used);
                const long over = T - (long) BLOCK - (long) pad*BLOCK;
                if (over > 0) {
                    printf("  %s fb=%d n_used=%d n_exp=%ld ne12=%ld: rows=%ld J=%d nthreads=%d tile reads %ld B, pad %d blocks = %ld B -> %ld B past\n",
                           PN[pi], fb, n_used, (long) ne02, (long) ne12, (long) (ne12*n_used), J, nth, T, pad, (long) pad*BLOCK, over);
                }
            }
        }
    }
    return 0;
}
