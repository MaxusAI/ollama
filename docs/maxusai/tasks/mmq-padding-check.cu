// GPU-free check of the MMQ tail padding in ggml_cuda_mul_mat_q(), for every padding rule proposed so far.
// Material for a successor to llama.cpp#27044 on master dd266785c (#29941 merged); see
// docs/maxusai/upstream-mmq-successor-material.md. It extends mmq-ids-padding-test.cu (#445) in three ways:
// one expert per token (n_expert_used = 1), the ids_dst read, and the non-ids (dense) branch.
//
// What the kernel reads (mmq.cuh at dd266785c):
//   - src1: a tile loads J blocks of block_q8_1_mmq from its first column, with no bound
//     (`for (l0 < J*MMQ_TILE_Y_K) tile_y[l] = by0[l]`). The buffer is laid out [k-plane][column], so in the last
//     k-plane a tile holding r valid columns reads J - r blocks past the data. The padding must cover that.
//       ids branch: the last expert's last tile can hold a single row, for any batch, so the padding must be at
//                   least J - 1 blocks.
//       dense branch: the last tile holds ne11 - (ntiles - 1)*J columns, so the padding must be at least
//                   ntiles*J - ne11 blocks.
//   - ids_dst: each tile loads J entries, `ids_dst_shared[j] = ids_dst[col_low + jt*J + j]` for j < J, again with no
//     bound, so the last tile reads up to J - 1 entries past ids_dst's ne12*n_expert_used. Only the successor's rule
//     pads it.
// J is the tile width mul_mat_q_switch_J launches: the smallest J (8..128, step 8, a valid config that fits shared
// memory) with the fewest tiles for ncols_opt. That loop is replicated below because it sits in a template that
// queries the device; everything else is llama.cpp's own code.
//
// The ids-branch rules (the argument of ggml_cuda_mmq_get_J_max(), or the successor's widest tile):
//   #24127    : ne11                (1 for broadcast gate/up, n_expert_used otherwise), b9992 .. dd266785c^
//   master    : ne12                (#29941, merged as dd266785c)
//   #27044    : ne12*n_expert_used  (the fork's compat 903)
//   successor : the widest tile that has a config, get_J_max(type, fallback, cc, 512), for src1 and for ids_dst.
//               Before #24127 (b9990) both branches padded by the widest tile, get_mmq_x_max_host(cc).
//
// Build from the root of a llama.cpp checkout, with the CUDA toolkit's nvcc. It never touches a GPU:
//   nvcc -std=c++17 -I. -Iggml/include -Iggml/src -Iggml/src/ggml-cuda mmq-padding-check.cu -o mmq-padding-check
//   ./mmq-padding-check
// The exit code is non-zero only if the successor's rule leaves a shape short.
#include "ggml/src/ggml-cuda/mmq.cuh"
#include "ggml/src/ggml-cuda/mmvq.cuh"   // MMVQ_MAX_BATCH_SIZE
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { abort(); }
int ggml_cuda_get_device() { return 0; }  // never reached: nothing below queries a device
extern "C" void ggml_abort(const char *, int, const char *, ...) { abort(); }

// The largest MUL_MAT_ID batch MMVQ takes (mmvq.cu, get_mmvq_mmid_max_batch_*; defined outside its header, so
// replicated for the ten types checked). A larger batch goes to MMQ.
static int mmvq_mmid_max(ggml_type t, int cc) {
    if (GGML_CUDA_CC_IS_RDNA3(cc)) {
        switch (t) {
            case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: return 4;
            default: return MMVQ_MAX_BATCH_SIZE;
        }
    }
    switch (t) {  // Turing and newer
        case GGML_TYPE_Q2_K: return 7;
        case GGML_TYPE_Q3_K: return 5;
        default:             return MMVQ_MAX_BATCH_SIZE;
    }
}

// mul_mat_q_switch_J's choice. Returns the launched tile width, 0 if none.
static int launched_J(ggml_type t, bool fb, int cc, size_t smpbo, int64_t ncols_opt) {
    int nt_best = INT_MAX, J_launch = 0;
    for (int J = 8; J <= 128 && nt_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fb, cc, GGML_PREC_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int nt = (ncols_opt + c.J - 1) / c.J;
        if (nt < nt_best) { nt_best = nt; J_launch = c.J; }
    }
    return J_launch;
}

static const char * const ARCH_NAMES[5] = {"sm_75", "sm_86", "sm_89", "sm_120", "gfx1151"};
static int arch_index(const char * name) { for (int i = 0; i < 5; ++i) if (!strcmp(name, ARCH_NAMES[i])) return i; return 0; }

struct tally {
    long n = 0; long n_arch[5] = {}; int worst = 0; char ex[200] = "";
    void add(int shortfall, const char * fmt_arch, const char * tname, int fb, long a, long b, long c, int J, int pad) {
        if (shortfall <= 0) return;
        ++n; ++n_arch[arch_index(fmt_arch)];
        if (shortfall > worst) {
            worst = shortfall;
            snprintf(ex, sizeof(ex), "%s, %s, fallback=%d, %ld/%ld/%ld: J=%d, padding=%d", fmt_arch, tname, fb, a, b, c, J, pad);
        }
    }
    void print(const char * label, const char * unit) const {
        printf("  %-44s %s", label, n ? "SHORT" : "covered");
        if (n) {
            printf(" in %ld shapes, worst %d %s past the allocation (%s)\n", n, worst, unit, ex);
            printf("  %-44s  by arch:", "");
            for (int i = 0; i < 5; ++i) printf(" %s %ld%s", ARCH_NAMES[i], n_arch[i], i < 4 ? "," : "");
        }
        printf("\n");
    }
};

int main() {
    // smpbo = max shared memory per block (opt-in). sm_120 was read from an RTX PRO 6000 Blackwell; gfx1151 has 64 KiB.
    struct arch { const char * name; int cc; size_t smpbo; };
    const arch archs[] = {
        {"sm_75",   GGML_CUDA_CC_TURING,        65536},
        {"sm_86",   860,                       101376},
        {"sm_89",   GGML_CUDA_CC_ADA_LOVELACE, 101376},
        {"sm_120",  GGML_CUDA_CC_BLACKWELL,    101376},
        {"gfx1151", GGML_CUDA_CC_RDNA3_5,       65536},
    };
    const ggml_type types[] = {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                               GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
    const char * tn[] = {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K"};
    const int ne02s[] = {8, 32, 128, 256};   // experts; only RDNA3/4 optimise J against ne12*n_expert_used/ne02

    // ids branch: [rule][n_expert_used == 1 ? 0 : 1]
    enum { R24127, RMASTER, R27044, RSUCC, NR };
    const char * rn[NR] = {"#24127 (ne11)", "master = #29941 (ne12)", "#27044 (ne12*n_expert_used)", "successor (widest tile)"};
    tally ids_src1[NR][2], ids_dst[NR][2], dense[2];
    long ids_shapes = 0, dense_shapes = 0;

    for (const arch & a : archs) for (int ti = 0; ti < 10; ++ti) for (int fb = 0; fb < 2; ++fb) {
        const ggml_type t = types[ti];
        const bool rdna = GGML_CUDA_CC_IS_RDNA3(a.cc) || GGML_CUDA_CC_IS_RDNA4(a.cc);
        const int pad_widest = ggml_cuda_mmq_get_J_max(t, fb, a.cc, 512);   // as mmq-successor-fix.patch

        // ids branch
        for (int64_t ne12 = mmvq_mmid_max(t, a.cc) + 1; ne12 <= 1024; ++ne12)
        for (int n_used = 1; n_used <= 16; ++n_used)
        for (int ei = 0; ei < (rdna ? 4 : 1); ++ei) {
            const int64_t ne02 = rdna ? ne02s[ei] : 256;
            if (n_used > ne02) continue;
            const int64_t ncols_opt = rdna ? (ne12*n_used + ne02 - 1) / ne02 : ne12;
            const int J = launched_J(t, fb, a.cc, a.smpbo, ncols_opt);
            if (J == 0) continue;
            for (int bcast = 0; bcast < 2; ++bcast) {
                if (bcast && n_used == 1) continue;   // one expert per token has nothing to broadcast
                ++ids_shapes;
                const int pad[NR] = {
                    ggml_cuda_mmq_get_J_max(t, fb, a.cc, bcast ? 1 : n_used),
                    ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne12),
                    ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne12*n_used),
                    pad_widest,
                };
                const int u = n_used == 1 ? 0 : 1;
                for (int r = 0; r < NR; ++r) {
                    ids_src1[r][u].add((J - 1) - pad[r], a.name, tn[ti], fb, ne12, n_used, ne02, J, pad[r]);
                    const int dst_pad = r == RSUCC ? pad_widest : 0;   // only the successor pads ids_dst
                    ids_dst[r][u].add((J - 1) - dst_pad, a.name, tn[ti], fb, ne12, n_used, ne02, J, dst_pad);
                }
            }
        }

        // dense branch: MMVQ takes ne11 <= MMVQ_MAX_BATCH_SIZE
        for (int64_t ne11 = MMVQ_MAX_BATCH_SIZE + 1; ne11 <= 1024; ++ne11) {
            const int J = launched_J(t, fb, a.cc, a.smpbo, ne11);
            if (J == 0) continue;
            ++dense_shapes;
            const int past = (int) (((ne11 + J - 1) / J) * J - ne11);
            const int pad_master = ggml_cuda_mmq_get_J_max(t, fb, a.cc, ne11);
            dense[0].add(past - pad_master, a.name, tn[ti], fb, ne11, 0, 0, J, pad_master);
            dense[1].add(past - pad_widest, a.name, tn[ti], fb, ne11, 0, 0, J, pad_widest);
        }
    }

    printf("ids branch: %ld shapes (sm_75/86/89/120 and gfx1151 x 10 types x fallback 0/1 x every MMQ batch up to 1024 tokens\n"
           "            x n_expert_used 1..16 x broadcast 0/1; on gfx1151 also x 8/32/128/256 experts)\n", ids_shapes);
    printf(" src1_q8_1, one expert per token (n_expert_used = 1):\n");
    for (int r = 0; r < NR; ++r) ids_src1[r][0].print(rn[r], "blocks");
    printf(" src1_q8_1, two or more experts per token:\n");
    for (int r = 0; r < NR; ++r) ids_src1[r][1].print(rn[r], "blocks");
    printf(" ids_dst, any n_expert_used:\n");
    for (int r = 0; r < NR; ++r) {
        tally both = ids_dst[r][0].n ? ids_dst[r][0] : ids_dst[r][1];
        both.n = ids_dst[r][0].n + ids_dst[r][1].n;
        for (int i = 0; i < 5; ++i) both.n_arch[i] = ids_dst[r][0].n_arch[i] + ids_dst[r][1].n_arch[i];
        if (ids_dst[r][1].worst > both.worst) { both.worst = ids_dst[r][1].worst; snprintf(both.ex, sizeof(both.ex), "%s", ids_dst[r][1].ex); }
        both.print(rn[r], "int32 entries");
    }
    printf("dense branch: %ld shapes (same archs, types and fallbacks x every MMQ batch up to 1024 columns)\n", dense_shapes);
    dense[0].print("master (ne11)", "blocks");
    dense[1].print("widest tile", "blocks");

    const bool succ_ok = ids_src1[RSUCC][0].n == 0 && ids_src1[RSUCC][1].n == 0 && ids_dst[RSUCC][0].n == 0 && ids_dst[RSUCC][1].n == 0;
    return succ_ok ? 0 : 1;
}
